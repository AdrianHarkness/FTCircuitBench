"""FTCircuitBench -> Azure Quantum Resource Estimator (QRE) bridge.

FTCircuitBench stops at the *logical* layer: Clifford+T gate counts and PBC
rotation/measurement counts. This module carries those counts down to the
*physical* layer -- physical qubits, wall-clock runtime, code distance, and
T-factory count -- by feeding them to the Azure QRE that ships inside the `qdk`
package. The estimator runs locally; no Azure subscription or network access is
required.

`qdk` is an optional dependency:

    uv sync --extra qre          # or: pip install "ftcircuitbench[qre]"

Two notes on what the estimates mean:

* FTCircuitBench has already synthesised every `rz` into Clifford+T, so the
  counts handed to QRE carry a concrete `tCount` and leave `rotationCount` at
  zero. The physical cost therefore reflects *FTCircuitBench's* synthesis
  (Gridsynth or Solovay-Kitaev at the requested precision) rather than QRE's
  own internal rotation-cost model.
* QRE needs at least one T state to lay out a T factory; it fails on
  `tCount == 0`. The readers below drop purely Clifford circuits by default,
  and :func:`estimate_circuit` reports any that slip through as a per-model
  error rather than letting QRE's internal `KeyError` surface.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CircuitResourceEstimate",
    "DEFAULT_MODEL_NAMES",
    "HARDWARE_MODELS",
    "HardwareModel",
    "LogicalResourceCounts",
    "PhysicalEstimate",
    "QRE_INSTALL_HINT",
    "default_hardware_models",
    "estimate_circuit",
    "estimate_circuits",
    "is_qre_available",
    "logical_counts_from_stats",
    "qubits_from_label",
    "read_ct_stats_csv",
    "read_stats_json",
    "resolve_hardware_models",
]

QRE_INSTALL_HINT = (
    "Azure QRE support requires the optional `qdk` dependency. Install it with "
    '`uv sync --extra qre` or `pip install "ftcircuitbench[qre]"`.'
)

# Column names in circuit_benchmarks/ct_stats.csv (written by generate_benchmarks.py).
_CT_LABEL_COLUMN = "label"
_CT_T_COLUMN = "t gates"

_LABEL_QUBIT_RE = re.compile(r"(\d+)q")


# --- Hardware models ---------------------------------------------------------


@dataclass(frozen=True)
class HardwareModel:
    """One Azure QRE hardware configuration.

    Attributes:
        name: Human-readable key used in output dictionaries and CLI flags.
        qubit_params: Built-in Azure QRE qubit parameter set, e.g.
            ``"qubit_gate_ns_e3"``.
        qec_scheme: Built-in QEC scheme name (``"surface_code"``,
            ``"floquet_code"``). ``None`` leaves QRE's default for the qubit
            model in place.
        error_budget: Total error budget for the algorithm, split by QRE across
            logical errors, T-state distillation, and rotation synthesis.
    """

    name: str
    qubit_params: str
    qec_scheme: Optional[str] = None
    error_budget: float = 0.01

    def job_params(self) -> Dict[str, Any]:
        """Render this model as an Azure QRE job-parameter dictionary."""
        params: Dict[str, Any] = {
            "qubitParams": {"name": self.qubit_params},
            "errorBudget": self.error_budget,
        }
        if self.qec_scheme is not None:
            params["qecScheme"] = {"name": self.qec_scheme}
        return params

    def with_error_budget(self, error_budget: float) -> "HardwareModel":
        """Return a copy of this model with a different error budget."""
        return HardwareModel(
            name=self.name,
            qubit_params=self.qubit_params,
            qec_scheme=self.qec_scheme,
            error_budget=error_budget,
        )


# Azure QRE's six built-in qubit parameter sets. Majorana models are paired with
# the floquet code, which is the QEC scheme those parameter sets are published
# with; the gate-based models use QRE's default surface code.
HARDWARE_MODELS: Dict[str, HardwareModel] = {
    model.name: model
    for model in (
        HardwareModel("superconducting 1e-3", "qubit_gate_ns_e3"),
        HardwareModel("superconducting 1e-4", "qubit_gate_ns_e4"),
        HardwareModel("trapped-ion 1e-3", "qubit_gate_us_e3"),
        HardwareModel("trapped-ion 1e-4", "qubit_gate_us_e4"),
        HardwareModel("majorana 1e-4", "qubit_maj_ns_e4", qec_scheme="floquet_code"),
        HardwareModel("majorana 1e-6", "qubit_maj_ns_e6", qec_scheme="floquet_code"),
    )
}

DEFAULT_MODEL_NAMES: Tuple[str, ...] = (
    "superconducting 1e-3",
    "superconducting 1e-4",
)


def default_hardware_models() -> List[HardwareModel]:
    """The models used when a caller doesn't ask for a specific set."""
    return [HARDWARE_MODELS[name] for name in DEFAULT_MODEL_NAMES]


def resolve_hardware_models(
    names: Optional[Iterable[str]] = None,
    error_budget: Optional[float] = None,
) -> List[HardwareModel]:
    """Look up hardware models by name, optionally overriding the error budget.

    Args:
        names: Model names from :data:`HARDWARE_MODELS`. ``None`` selects
            :data:`DEFAULT_MODEL_NAMES`.
        error_budget: If given, applied to every resolved model.

    Raises:
        KeyError: If a requested name is not a known model.
    """
    models = (
        default_hardware_models()
        if names is None
        else [_lookup_model(name) for name in names]
    )
    if error_budget is not None:
        models = [model.with_error_budget(error_budget) for model in models]
    return models


def _lookup_model(name: str) -> HardwareModel:
    try:
        return HARDWARE_MODELS[name]
    except KeyError:
        raise KeyError(
            f"Unknown hardware model '{name}'. "
            f"Available models: {', '.join(sorted(HARDWARE_MODELS))}."
        ) from None


# --- Logical counts ----------------------------------------------------------


@dataclass(frozen=True)
class LogicalResourceCounts:
    """Logical-layer counts for one circuit, ready to hand to Azure QRE.

    Mirrors the fields of QRE's ``LogicalCounts`` input, minus the rotation
    fields that FTCircuitBench has already discharged during synthesis.
    """

    label: str
    num_qubits: int
    t_count: int
    measurement_count: int
    ccz_count: int = 0
    counts_source: str = "clifford_t"

    def to_job_input(self) -> Dict[str, int]:
        """Render as the dictionary QRE's ``LogicalCounts`` accepts."""
        job_input = {
            "numQubits": self.num_qubits,
            "tCount": self.t_count,
            "measurementCount": self.measurement_count,
        }
        if self.ccz_count:
            job_input["cczCount"] = self.ccz_count
        return job_input


def qubits_from_label(label: str) -> int:
    """Extract the qubit count encoded in a benchmark label.

    Benchmark labels carry their width as an ``<N>q`` token, e.g.
    ``"adder-10q-gs-5"`` -> 10, ``"heisenberg-2d-tri-100q-sk-2"`` -> 100.

    Raises:
        ValueError: If the label carries no ``<N>q`` token.
    """
    match = _LABEL_QUBIT_RE.search(label)
    if match is None:
        raise ValueError(
            f"Cannot infer qubit count from label '{label}': expected an "
            "'<N>q' token such as '10q'."
        )
    return int(match.group(1))


def _parse_count(raw: str, column: str, label: str) -> int:
    """Parse an integer cell, tolerating thousands separators and padding."""
    cleaned = raw.strip().replace(",", "") if raw is not None else ""
    try:
        return int(cleaned)
    except ValueError:
        raise ValueError(
            f"Row '{label}' has a non-integer '{column}' value: {raw!r}."
        ) from None


def read_ct_stats_csv(
    path: str = os.path.join("circuit_benchmarks", "ct_stats.csv"),
    labels: Optional[Iterable[str]] = None,
    skip_zero_t: bool = True,
) -> List[LogicalResourceCounts]:
    """Read logical counts from an aggregated Clifford+T stats CSV.

    The CSV is the one `generate_benchmarks.py` writes to
    `circuit_benchmarks/ct_stats.csv`: one row per circuit/pipeline pair, with
    the circuit width encoded in the label. `measurement_count` is set to the
    qubit count, i.e. one terminal measurement per qubit.

    Args:
        path: Path to the CSV.
        labels: Optional allow-list of labels; rows outside it are dropped.
        skip_zero_t: Drop Clifford-only rows, which QRE cannot estimate.

    Raises:
        ValueError: If a required column is missing or a row is malformed.
    """
    wanted = None if labels is None else set(labels)
    counts: List[LogicalResourceCounts] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [c for c in (_CT_LABEL_COLUMN, _CT_T_COLUMN) if c not in fieldnames]
        if missing:
            raise ValueError(
                f"{path} is missing required column(s) {missing}. Found: {fieldnames}."
            )
        for row in reader:
            label = (row.get(_CT_LABEL_COLUMN) or "").strip()
            if not label or (wanted is not None and label not in wanted):
                continue
            num_qubits = qubits_from_label(label)
            t_count = _parse_count(row[_CT_T_COLUMN], _CT_T_COLUMN, label)
            if skip_zero_t and t_count == 0:
                continue
            counts.append(
                LogicalResourceCounts(
                    label=label,
                    num_qubits=num_qubits,
                    t_count=t_count,
                    measurement_count=num_qubits,
                )
            )
    return counts


def logical_counts_from_stats(
    stats: Mapping[str, Any],
    label: str,
    counts_source: str = "clifford_t",
) -> LogicalResourceCounts:
    """Build logical counts from a per-circuit stats dictionary.

    Accepts the dictionaries written to `circuit_stats_output/*_stats.json` by
    `analyze_circuit.py` (equivalently, a merged `PipelineResult.clifford_stats`
    / `pbc_stats` pair).

    Args:
        stats: The stats mapping.
        label: Label to attach to the resulting counts.
        counts_source: `"clifford_t"` uses the synthesised Clifford+T T-family
            count; `"pbc"` uses the post-optimization PBC rotation and
            measurement operator counts instead, which is the cost model that
            applies when the circuit is executed as Pauli-based computation.

    Raises:
        KeyError: If the stats mapping lacks the keys the source needs.
        ValueError: If `counts_source` is not a recognised source.
    """
    if counts_source not in ("clifford_t", "pbc"):
        raise ValueError(
            f"Unknown counts_source '{counts_source}'; expected 'clifford_t' or 'pbc'."
        )

    num_qubits = _require_int(stats, "num_qubits", label)
    if counts_source == "clifford_t":
        t_count = _require_int(stats, "total_t_family_count", label)
        measurement_count = num_qubits
    else:
        # Older stats files record the post-optimization rotation count under
        # `pbc_t_operators` only; the two keys agree where both are present.
        t_count = _require_int(
            stats, "pbc_rotation_operators", label, fallback_key="pbc_t_operators"
        )
        measurement_count = int(stats.get("pbc_measurement_operators", num_qubits))

    return LogicalResourceCounts(
        label=label,
        num_qubits=num_qubits,
        t_count=t_count,
        measurement_count=measurement_count,
        counts_source=counts_source,
    )


def _require_int(
    stats: Mapping[str, Any],
    key: str,
    label: str,
    fallback_key: Optional[str] = None,
) -> int:
    if key not in stats and fallback_key is not None and fallback_key in stats:
        key = fallback_key
    if key not in stats:
        raise KeyError(
            f"Stats for '{label}' have no '{key}' entry; cannot build logical counts."
        )
    value = stats[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise KeyError(f"Stats for '{label}' have a non-numeric '{key}': {value!r}.")
    return int(value)


def read_stats_json(
    path: str,
    label: Optional[str] = None,
    counts_source: str = "clifford_t",
) -> LogicalResourceCounts:
    """Read logical counts from one `*_stats.json` file.

    Args:
        path: Path to a stats JSON written by `analyze_circuit.py`.
        label: Label to attach; defaults to the filename with the trailing
            `_stats` suffix stripped.
        counts_source: See :func:`logical_counts_from_stats`.
    """
    with open(path, encoding="utf-8") as handle:
        stats = json.load(handle)
    if label is None:
        stem = os.path.splitext(os.path.basename(path))[0]
        label = stem[: -len("_stats")] if stem.endswith("_stats") else stem
    return logical_counts_from_stats(stats, label, counts_source=counts_source)


# --- Estimation --------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalEstimate:
    """Physical-layer estimate for one circuit under one hardware model."""

    model: str
    physical_qubits: int
    runtime_seconds: float
    code_distance: int
    logical_depth: int
    algorithmic_logical_qubits: int
    num_t_states: int
    num_t_factories: int
    physical_qubits_for_algorithm: int
    physical_qubits_for_t_factories: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "physical_qubits": self.physical_qubits,
            "runtime_s": self.runtime_seconds,
            "code_distance": self.code_distance,
            "logical_depth": self.logical_depth,
            "algorithmic_logical_qubits": self.algorithmic_logical_qubits,
            "num_t_states": self.num_t_states,
            "num_t_factories": self.num_t_factories,
            "physical_qubits_for_algorithm": self.physical_qubits_for_algorithm,
            "physical_qubits_for_t_factories": self.physical_qubits_for_t_factories,
        }


@dataclass(frozen=True)
class CircuitResourceEstimate:
    """Estimates for one circuit across the requested hardware models."""

    counts: LogicalResourceCounts
    estimates: Dict[str, PhysicalEstimate] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.counts.label

    def to_dict(self) -> Dict[str, Any]:
        models: Dict[str, Any] = {
            name: estimate.to_dict() for name, estimate in self.estimates.items()
        }
        models.update({name: {"error": msg} for name, msg in self.errors.items()})
        return {
            "label": self.counts.label,
            "num_qubits": self.counts.num_qubits,
            "t_count": self.counts.t_count,
            "measurement_count": self.counts.measurement_count,
            "counts_source": self.counts.counts_source,
            "models": models,
        }


def is_qre_available() -> bool:
    """True when the optional Azure QRE dependency can be imported."""
    try:
        _import_logical_counts()
        return True
    except ImportError:
        return False


def _import_logical_counts() -> Any:
    """Import QRE's `LogicalCounts`, preferring `qdk` over legacy `qsharp`."""
    try:
        from qdk.estimator import LogicalCounts

        return LogicalCounts
    except ImportError:
        pass
    try:
        # `qsharp` predates the rename to `qdk`; recent releases are a shim.
        from qsharp.estimator import LogicalCounts as LegacyLogicalCounts

        return LegacyLogicalCounts
    except ImportError as exc:
        raise ImportError(QRE_INSTALL_HINT) from exc


def _to_physical_estimate(
    model_name: str, result: Mapping[str, Any]
) -> PhysicalEstimate:
    """Pull the fields we report out of a QRE result object."""
    physical = result["physicalCounts"]
    breakdown = physical["breakdown"]
    return PhysicalEstimate(
        model=model_name,
        physical_qubits=int(physical["physicalQubits"]),
        # QRE reports runtime in nanoseconds.
        runtime_seconds=float(physical["runtime"]) / 1e9,
        code_distance=int(result["logicalQubit"]["codeDistance"]),
        logical_depth=int(breakdown["logicalDepth"]),
        algorithmic_logical_qubits=int(breakdown["algorithmicLogicalQubits"]),
        num_t_states=int(breakdown["numTstates"]),
        num_t_factories=int(breakdown["numTfactories"]),
        physical_qubits_for_algorithm=int(breakdown["physicalQubitsForAlgorithm"]),
        physical_qubits_for_t_factories=int(breakdown["physicalQubitsForTfactories"]),
    )


def estimate_circuit(
    counts: LogicalResourceCounts,
    models: Optional[Sequence[HardwareModel]] = None,
) -> CircuitResourceEstimate:
    """Estimate physical resources for one circuit under each hardware model.

    A model whose estimate fails (an infeasible error budget, say) is recorded
    in `errors` rather than aborting the remaining models.

    Raises:
        ImportError: If the optional Azure QRE dependency is not installed.
    """
    logical_counts_cls = _import_logical_counts()
    selected = list(models) if models is not None else default_hardware_models()

    estimates: Dict[str, PhysicalEstimate] = {}
    errors: Dict[str, str] = {}
    if counts.t_count <= 0:
        # QRE needs a T factory to lay out and fails deep inside its result
        # parsing when there are no T states. Report it as a per-model error
        # rather than letting that surface as a bare KeyError.
        message = (
            "Azure QRE requires tCount > 0; "
            f"'{counts.label}' has no T gates to distill."
        )
        return CircuitResourceEstimate(
            counts=counts, errors={model.name: message for model in selected}
        )

    job_input = counts.to_job_input()
    for model in selected:
        try:
            result = logical_counts_cls(job_input).estimate(model.job_params())
            estimates[model.name] = _to_physical_estimate(model.name, result)
        except Exception as exc:  # QRE raises library-specific error types
            errors[model.name] = f"{type(exc).__name__}: {exc}"
    return CircuitResourceEstimate(counts=counts, estimates=estimates, errors=errors)


def estimate_circuits(
    counts: Iterable[LogicalResourceCounts],
    models: Optional[Sequence[HardwareModel]] = None,
    progress: bool = False,
) -> List[CircuitResourceEstimate]:
    """Estimate physical resources for several circuits.

    Args:
        counts: Logical counts, e.g. from :func:`read_ct_stats_csv`.
        models: Hardware models; defaults to :func:`default_hardware_models`.
        progress: Print a one-line-per-circuit progress trace.
    """
    results: List[CircuitResourceEstimate] = []
    for item in counts:
        if progress:
            print(f"  estimating {item.label} (T={item.t_count:,})")
        results.append(estimate_circuit(item, models=models))
    return results
