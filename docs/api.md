# API Reference

This page documents the public API surface re-exported from the top-level
`ftcircuitbench` package and the higher-level helpers in `ftcircuitbench.api`.
All symbols listed here appear in [`ftcircuitbench/__init__.py`](../ftcircuitbench/__init__.py)'s
`__all__`.

```python
import ftcircuitbench as ftb
# or, for the high-level pipeline helpers
from ftcircuitbench.api import PipelineConfig, run_analysis_for_file
```

The supported entry points are:

- the dataclasses and functions in [`ftcircuitbench/api.py`](../ftcircuitbench/api.py)
  (`PipelineConfig`, `PipelineResult`, `AnalysisResult`, `run_pipeline`,
  `run_analysis`, `run_analysis_for_file`);
- the lower-level building blocks (parser, decomposer, transpilers, PBC
  converter, fidelity, analyzers, visualization helpers) re-exported from the
  package root for users who want to assemble custom pipelines.

---

## Pipeline configuration and results

These live in [`ftcircuitbench/api.py`](../ftcircuitbench/api.py).

### `PipelineConfig`

```python
@dataclass
class PipelineConfig:
    pipeline: Literal["gs", "sk"] = "gs"
    gridsynth_precision: int = 3
    sk_recursion: int = 1
    layering_method: Literal["bare", "v2", "singleton"] = "v2"
    layering_max_checks: Optional[int] = None
    optimize_pbc: bool = False
    optimize_t_maxiter: int = 5
    prefer_cpp: bool = True
    calculate_fidelity: bool = True
    return_intermediate: bool = True
    max_workers: Optional[int] = None
    clifford_output_path: Optional[str] = None
    pbc_output_prefix: Optional[str] = None
```

Configuration for a single pipeline run. Key fields:

- `pipeline` — `"gs"` for Gridsynth or `"sk"` for Solovay-Kitaev.
- `gridsynth_precision` / `sk_recursion` — pipeline-specific knobs.
- `layering_method` — PBC layering strategy:
  - `"bare"` — single-pass greedy grouping; fastest per-call but produces more layers.
  - `"v2"` — backward-scan grouping; default. Accepts `layering_max_checks` to
    bound the scan window.
  - `"singleton"` — one rotation per layer (disables grouping).
- `layering_max_checks` — bound on `v2`'s backward layer scan. `None` (the
  default) scans all prior layers. Ignored for `"bare"` and `"singleton"`.
- `optimize_pbc` / `optimize_t_maxiter` — toggle and bound the PBC T-merging
  optimisation.
- `prefer_cpp` — prefer the nwqec C++ backend when available; otherwise the
  Python fallback is used.
- `calculate_fidelity` — compute fidelity between the input and the Clifford+T
  circuit when feasible. Fidelity is automatically skipped for SK pipelines on
  circuits with more than `MAX_QUBITS_FOR_FIDELITY` qubits.
- `clifford_output_path` / `pbc_output_prefix` — when set, persist the
  Clifford+T QASM and PBC layer/measurement-basis text files.

### `PipelineResult`

```python
@dataclass
class PipelineResult:
    pipeline: PipelineName
    clifford_t_circuit: QuantumCircuit
    pbc_circuit: QuantumCircuit
    clifford_stats: Dict[str, Any]
    pbc_stats: Dict[str, Any]
    fidelity: Optional[Dict[str, Any]]
    timings: Dict[str, float]
    parameters: Dict[str, Any]
    intermediate_circuit: Optional[QuantumCircuit] = None
    artifacts: Optional[Dict[str, str]] = None
```

Structured result returned by `run_pipeline`. Contains the compiled circuits,
analyzer outputs, optional fidelity result, per-stage timings, and the paths of
any artefacts written to disk.

`PipelineResult.to_dict(include_circuits=False, include_artifacts=True)` —
return a JSON-serialisable view; set `include_circuits=True` to embed the
Clifford+T and PBC QASM strings in the dict.

### `AnalysisResult`

```python
@dataclass
class AnalysisResult:
    input_path: Optional[str]
    original_qubits: int
    original_gates: int
    pipelines: Dict[str, PipelineResult]
```

Aggregated result for one or more pipelines run on a single input circuit.
`AnalysisResult.to_dict(include_circuits=False, include_artifacts=True)` mirrors
the `PipelineResult` accessor and applies it to every pipeline in `pipelines`.

---

## Pipeline entry points

### `run_pipeline(circuit, config) -> PipelineResult`

Execute a single pipeline (Clifford+T synthesis, PBC conversion, optional
fidelity) on the provided `QuantumCircuit`.

### `run_analysis(circuit, configs, source_path=None) -> AnalysisResult`

Run one or more `PipelineConfig` objects on an in-memory circuit. `configs` may
be a single `PipelineConfig` or a sequence; results are keyed by the
`pipeline` field of each config.

### `run_analysis_for_file(qasm_file, configs) -> AnalysisResult`

Convenience wrapper around `load_qasm_circuit` followed by `run_analysis`. The
resulting `AnalysisResult.input_path` is set to `qasm_file`.

---

## Parser helpers

Defined in [`ftcircuitbench/parser/qasm_parser.py`](../ftcircuitbench/parser/qasm_parser.py).

### `load_qasm_circuit(qasm_input, is_file=True) -> QuantumCircuit`

Load a quantum circuit from a QASM file or QASM string. Detects OpenQASM 2.0
vs 3.0 automatically, strips unsupported `reset` instructions with a warning,
and rebuilds the circuit on a single quantum register so the result is
compatible with the rest of the pipeline.

### `transpile_qasm_to_target_basis(qasm_input, is_file=True, basis_gates=None, optimization_level=0) -> QuantumCircuit`

Load and transpile a QASM input into the FTCircuitBench default target basis
(or a user-supplied one). Optimisation level defaults to 0 to preserve a clean
baseline circuit.

---

## Decomposer

Defined in [`ftcircuitbench/decomposer/decomposer.py`](../ftcircuitbench/decomposer/decomposer.py).

### `decompose_rz_gates_gridsynth(original_circuit, precision=10, progress_bar=None) -> QuantumCircuit`

Decompose every `Rz` gate in a circuit into `S`, `H`, `T` (and possibly `X`)
gates using the `gridsynth` CLI. Identity rotations are dropped; non-numeric
parameter expressions raise a `ValueError`. Requires the `gridsynth` binary on
`PATH`.

---

## Transpilers

Defined in [`ftcircuitbench/transpilers/__init__.py`](../ftcircuitbench/transpilers/__init__.py).

### `transpile_to_gridsynth_clifford_t(circuit_input, *, is_file=False, gridsynth_precision=3, remove_final_measurements=True, return_intermediate=False, prefer_cpp=True, force_python=False)`

Transpile a circuit (or QASM input) into Clifford+T using Gridsynth Rz
decomposition. Uses the high-performance `nwqec` C++ backend when available
and falls back to the pure-Python implementation otherwise. With
`return_intermediate=True` the function returns `(intermediate, final)` where
`intermediate` is the Clifford+Rz circuit prior to Rz decomposition.

### `transpile_to_solovay_kitaev_clifford_t(circuit, *, recursion_degree=3, remove_final_measurements=True, return_intermediate=False)`

Transpile a circuit into Clifford+T via Solovay-Kitaev. Python-only;
`nwqec` does not provide an SK backend.

---

## PBC converter

Defined in [`ftcircuitbench/pbc_converter/pbc_generator.py`](../ftcircuitbench/pbc_converter/pbc_generator.py).

### `convert_to_pbc_circuit(clifford_t_circuit, *, optimize_pbc=False, optimize_t_maxiter=5, if_print_rpc=False, layering_method="v2", layering_max_checks=None, output_prefix=None, max_workers=None, use_nwqec=True) -> tuple[QuantumCircuit, dict]`

Unified entry point for PBC conversion. Uses the `nwqec` C++ backend when
available; otherwise falls back to the parallel Python rotation-Pauli circuit
implementation. Returns the PBC circuit alongside a stats dictionary that
includes timings, T-counts, and (when `output_prefix` is set) artefact paths.

---

## Fidelity

Defined in [`ftcircuitbench/fidelity.py`](../ftcircuitbench/fidelity.py).

### `MAX_QUBITS_FOR_FIDELITY`

Module-level constant (currently `7`) that bounds when the unitary-based
fidelity calculation is used; circuits above this bound use the per-`Rz`
product-fidelity method.

### `rz_product_fidelity(original_qc, gridsynth_precision, use_multiprocessing=True) -> Dict[str, Any]`

Estimate fidelity by computing the per-`Rz` decomposition fidelity and taking
the product over all `Rz` gates. Scales linearly with the number of `Rz`
gates rather than exponentially with the number of qubits.

### `calculate_circuit_fidelity(original_qc, decomposed_qc, gridsynth_precision, sk_recursion_degree=None, intermediate_qc=None) -> Dict[str, Any]`

Top-level fidelity helper. Selects the unitary-based method below
`MAX_QUBITS_FOR_FIDELITY` qubits and `rz_product_fidelity` above it. The
`intermediate_qc` argument is recommended whenever the original circuit
contains custom gates.

---

## Analyzers

Defined under [`ftcircuitbench/analyzer/`](../ftcircuitbench/analyzer).

### `analyze_clifford_t_circuit(circuit, gridsynth_precision_used=None) -> dict`

Analyse a Clifford+T circuit. Returned dictionary includes T/Tdg counts,
total T-family count, two-qubit interaction graph statistics, and (when
provided) the precision used during compilation.

### `analyze_pbc_circuit(pbc_circuit, pbc_conversion_stats=None) -> dict`

Analyse a PBC circuit. Returns operator-weight distributions, rotation /
measurement / utility op counts, multi-qubit interaction statistics, and
merges in any stats produced by `convert_to_pbc_circuit`.

---

## Visualization helpers

Defined in [`ftcircuitbench/analyzer/visualization.py`](../ftcircuitbench/analyzer/visualization.py).
All four plotting helpers take a circuit and render via Matplotlib; see the
source for the full list of styling kwargs.

### `show_clifford_t_interaction_graph(circuit, *, title=..., figsize=(14, 10), ...) -> None`

Network graph where nodes are qubits (coloured by interaction degree) and
edges represent two-qubit gate interactions (thickness proportional to count).

### `show_pbc_interaction_graph(circuit, *, title=..., figsize=(14, 10), ...) -> None`

Same idea as the Clifford+T interaction graph but for multi-qubit Pauli
operator interactions in a PBC circuit.

### `show_operator_weight_histogram(circuit, *, title=..., figsize=(10, 6), ...) -> None`

Bar plot of rotation- and measurement-operator Pauli weights in a PBC circuit.

### `show_qubit_pbc_operations_plot(circuit, *, title=..., figsize=(12, 6), ...) -> None`

Bar plot of the number of PBC rotation + measurement operations each qubit
participates in.

### `get_interaction_statistics(circuit) -> Dict[str, Any]`

Return interaction statistics (total two-qubit gates, qubit-degree map,
edge-count map, max degree, qubit count) for a circuit. Used by both the
analyzers and the visualization helpers.

---

## Frontends (`ftcircuitbench.frontends`)

Not re-exported from the package root: this subpackage needs the optional `cirq`
and `qualtran` extras (`uv sync --extra cirq --extra qualtran`).

```python
from ftcircuitbench.frontends import (
    bloq_to_qiskit, cirq_to_qiskit, bloq_t_complexity, count_measurement_uncompute,
)
```

### `is_cirq_available() -> bool` / `is_qualtran_available() -> bool`

Whether the corresponding optional dependency can be imported.

### `has_qasm_representation(op) -> bool`

True when Cirq can render an operation as OpenQASM 2 directly. This is the
stopping rule for decomposition — see `decompose_for_qasm`.

### `decompose_for_qasm(circuit, intercepting_decomposer=None) -> cirq.Circuit`

Decompose only as far as OpenQASM 2 requires. Unlike a bare `cirq.decompose`,
which lowers to a hardware gateset and rewrites `H`/`CNOT` into rotations, this
stops at the first QASM-expressible layer and preserves the generator's native
Clifford+T structure. `intercepting_decomposer` is an optional per-operation
rewrite applied before Cirq's own decomposition.

### `cirq_to_qasm2(circuit, decompose=True, allow_non_unitary=False, intercepting_decomposer=None) -> str`
### `cirq_to_qiskit(...) -> QuantumCircuit`

Lower a Cirq circuit to OpenQASM 2, or to a Qiskit circuit ready for
`run_pipeline`. Raises `NonUnitaryCircuitError` when mid-circuit measurement,
reset, or classical control survives decomposition — FTCircuitBench's Clifford+T
and PBC analysis models unitary circuits, and terminal measurements (which the
pipeline strips anyway) are the only exception. Cirq is big-endian and Qiskit
little-endian, so the converted circuit satisfies
`cirq.unitary(c) == Operator(qc.reverse_bits())` up to global phase; gate-to-qubit
assignment, and therefore every resource count, is preserved exactly.

### `op_census(circuit) -> Dict[str, int]`

Operation counts by gate type — a quick look at what a generator emitted.

### `bloq_to_cirq(bloq, unitary_uncompute=False, decompose=True) -> cirq.Circuit`
### `bloq_to_qasm2(...) -> str` / `bloq_to_qiskit(...) -> QuantumCircuit`

Lower a Qualtran Bloq. `unitary_uncompute=True` substitutes the unitary adjoint
of `And` for Qualtran's measurement-based one, which is what makes most
arithmetic and data-loading Bloqs analysable; it adds exactly 4 T gates per
substitution.

### `unitary_uncompute_interceptor(op)`

The substitution itself, usable as `decompose_for_qasm`'s
`intercepting_decomposer`. Returns the `And` compute circuit reversed and
inverted — exactly `And†` as a matrix — for a measurement-based `And` adjoint,
and `NotImplemented` for anything else.

### `bloq_t_complexity(bloq) -> Dict[str, Any]`
### `count_measurement_uncompute(bloq_or_circuit) -> int`

Qualtran's analytic cost model, and the number of measurement-based `And`
adjoints a Bloq contains. Together they predict what FTCircuitBench will measure:

```python
measured_t == bloq_t_complexity(bloq)["t"] + 4 * count_measurement_uncompute(bloq)
```

See [`examples.md`](examples.md#5-importing-circuits-from-qualtran-and-pyliqtr)
for worked usage and the pyLIQTR file-handoff workflow.

---

## Resource estimation (`ftcircuitbench.resource_estimation`)

Not re-exported from the package root: this subpackage needs the optional `qre`
extra (`uv sync --extra qre`), which installs the `qdk` package containing the
Azure Quantum Resource Estimator. Import it directly:

```python
from ftcircuitbench.resource_estimation import (
    HARDWARE_MODELS,
    estimate_circuit,
    estimate_circuits,
    logical_counts_from_stats,
    read_ct_stats_csv,
    read_stats_json,
    resolve_hardware_models,
)
```

### `is_qre_available() -> bool`

True when the optional Azure QRE dependency can be imported. `qdk` is preferred;
the legacy `qsharp` package is accepted as a fallback.

### `HardwareModel`

```python
@dataclass(frozen=True)
class HardwareModel:
    name: str
    qubit_params: str
    qec_scheme: Optional[str] = None
    error_budget: float = 0.01
```

One Azure QRE hardware configuration. `job_params()` renders it as QRE job
parameters; `with_error_budget(f)` returns a copy with a different budget.
`HARDWARE_MODELS` maps names to the six built-in configurations
(`superconducting 1e-3/1e-4`, `trapped-ion 1e-3/1e-4`, `majorana 1e-4/1e-6`);
`DEFAULT_MODEL_NAMES` holds the two superconducting defaults.

### `resolve_hardware_models(names=None, error_budget=None) -> list[HardwareModel]`

Look models up by name (defaulting to `DEFAULT_MODEL_NAMES`), optionally
overriding the error budget on all of them. Raises `KeyError` for unknown names.

### `LogicalResourceCounts`

```python
@dataclass(frozen=True)
class LogicalResourceCounts:
    label: str
    num_qubits: int
    t_count: int
    measurement_count: int
    ccz_count: int = 0
    counts_source: str = "clifford_t"
```

The logical-layer input to an estimate. `to_job_input()` renders it as QRE's
`LogicalCounts` dictionary.

### `read_ct_stats_csv(path="circuit_benchmarks/ct_stats.csv", labels=None, skip_zero_t=True) -> list[LogicalResourceCounts]`

Read logical counts from the aggregated Clifford+T stats CSV written by
`generate_benchmarks.py`. Circuit width comes from the `<N>q` token in each
label; `measurement_count` is set to that width. Clifford-only rows are dropped
unless `skip_zero_t=False`.

### `logical_counts_from_stats(stats, label, counts_source="clifford_t") -> LogicalResourceCounts`

Build counts from a per-circuit stats mapping (a `*_stats.json` payload, or a
merged `clifford_stats` / `pbc_stats` pair). `counts_source="clifford_t"` uses
`total_t_family_count`; `"pbc"` uses the post-optimization
`pbc_rotation_operators` and `pbc_measurement_operators`.
`read_stats_json(path, label=None, counts_source=...)` is the file-reading
wrapper, defaulting the label to the filename minus its `_stats` suffix.

### `estimate_circuit(counts, models=None) -> CircuitResourceEstimate`

Estimate physical resources for one circuit under each hardware model. A model
whose estimate fails is recorded in `.errors` instead of aborting the rest.
Raises `ImportError` (carrying an install hint) when the `qre` extra is absent.
`estimate_circuits(counts, models=None, progress=False)` is the batch form.

### `CircuitResourceEstimate` / `PhysicalEstimate`

`CircuitResourceEstimate` holds the input `counts`, an `estimates` dict keyed by
model name, and an `errors` dict for models that failed; `to_dict()` renders the
JSON shape written by `estimate_resources.py`. Each `PhysicalEstimate` carries
`physical_qubits`, `runtime_seconds`, `code_distance`, `logical_depth`,
`algorithmic_logical_qubits`, `num_t_states`, `num_t_factories`,
`physical_qubits_for_algorithm`, and `physical_qubits_for_t_factories`.

See [`examples.md`](examples.md#4-physical-resource-estimation-estimate_resourcespy)
for worked usage.

---

## Worked example

```python
from ftcircuitbench.api import PipelineConfig, run_analysis_for_file

cfgs = [
    PipelineConfig(pipeline="gs", gridsynth_precision=4, optimize_pbc=True),
    PipelineConfig(pipeline="sk", sk_recursion=2, calculate_fidelity=False),
]
analysis = run_analysis_for_file("qasm/qft/qft_4q.qasm", cfgs)
print(analysis.pipelines["gs"].clifford_stats["total_t_family_count"])
print(analysis.to_dict(include_artifacts=True))
```

For a longer end-to-end walkthrough see [`examples.md`](examples.md) and the
notebook `demos/FTCircuitBench_Pipeline_Demo.ipynb`.
