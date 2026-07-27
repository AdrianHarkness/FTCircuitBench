from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from ftcircuitbench.resource_estimation import azure_qre

CSV_HEADER = "label,total gates,depth,clifford gates,t gates\n"


def _write_ct_stats(tmp_path: Path, rows: str, name: str = "ct_stats.csv") -> Path:
    path = tmp_path / name
    path.write_text(CSV_HEADER + rows, encoding="utf-8")
    return path


# --- Label parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("adder-10q-gs-5", 10),
        ("qft-4q-sk-2", 4),
        ("heisenberg-2d-tri-100q-sk-2", 100),
        ("fermi-hubbard-1d-128q-gs-8", 128),
        ("qpe-H2-0-6-12q-gs-5", 12),
    ],
)
def test_qubits_from_label(label: str, expected: int):
    assert azure_qre.qubits_from_label(label) == expected


def test_qubits_from_label_rejects_label_without_width():
    with pytest.raises(ValueError, match="Cannot infer qubit count"):
        azure_qre.qubits_from_label("adder-gs-5")


# --- CSV reader --------------------------------------------------------------


def test_read_ct_stats_csv_builds_counts(tmp_path: Path):
    path = _write_ct_stats(
        tmp_path,
        "adder-10q-gs-5,142,99,86,56\nqft-4q-sk-1,300,120,244,1234\n",
    )
    counts = azure_qre.read_ct_stats_csv(str(path))

    assert [c.label for c in counts] == ["adder-10q-gs-5", "qft-4q-sk-1"]
    first = counts[0]
    assert (first.num_qubits, first.t_count, first.measurement_count) == (10, 56, 10)
    assert first.counts_source == "clifford_t"


def test_read_ct_stats_csv_skips_clifford_only_rows(tmp_path: Path):
    # QRE cannot lay out a T factory for tCount == 0, so those rows are dropped.
    path = _write_ct_stats(
        tmp_path, "adder-10q-gs-5,142,99,142,0\nqft-4q-sk-1,300,120,244,12\n"
    )
    assert [c.label for c in azure_qre.read_ct_stats_csv(str(path))] == ["qft-4q-sk-1"]
    kept = azure_qre.read_ct_stats_csv(str(path), skip_zero_t=False)
    assert [c.label for c in kept] == ["adder-10q-gs-5", "qft-4q-sk-1"]


def test_read_ct_stats_csv_filters_by_label(tmp_path: Path):
    path = _write_ct_stats(
        tmp_path, "adder-10q-gs-5,142,99,86,56\nqft-4q-sk-1,300,120,244,12\n"
    )
    counts = azure_qre.read_ct_stats_csv(str(path), labels=["qft-4q-sk-1"])
    assert [c.label for c in counts] == ["qft-4q-sk-1"]


def test_read_ct_stats_csv_tolerates_thousands_separators(tmp_path: Path):
    path = _write_ct_stats(tmp_path, 'qft-63q-gs-8,1,1,1,"1,234,567"\n')
    assert azure_qre.read_ct_stats_csv(str(path))[0].t_count == 1234567


def test_read_ct_stats_csv_rejects_missing_column(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("label,total gates\nadder-10q-gs-5,142\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        azure_qre.read_ct_stats_csv(str(path))


def test_read_ct_stats_csv_rejects_non_integer_t_count(tmp_path: Path):
    path = _write_ct_stats(tmp_path, "adder-10q-gs-5,142,99,86,n/a\n")
    with pytest.raises(ValueError, match="non-integer 't gates' value"):
        azure_qre.read_ct_stats_csv(str(path))


def test_read_ct_stats_csv_reads_the_committed_benchmark_csv(repo_root: Path):
    csv_path = repo_root / "circuit_benchmarks" / "ct_stats.csv"
    if not csv_path.exists():
        pytest.skip("circuit_benchmarks/ct_stats.csv not present in this checkout")
    counts = azure_qre.read_ct_stats_csv(str(csv_path))
    assert counts, "expected at least one estimable row"
    assert all(c.t_count > 0 and c.num_qubits > 0 for c in counts)


# --- Stats JSON reader -------------------------------------------------------


STATS = {
    "num_qubits": 10,
    "total_t_family_count": 56,
    "pbc_rotation_operators": 32,
    "pbc_measurement_operators": 10,
}


def test_logical_counts_from_stats_clifford_t():
    counts = azure_qre.logical_counts_from_stats(STATS, "adder_10q_gs_prec5")
    assert (counts.t_count, counts.measurement_count) == (56, 10)
    assert counts.counts_source == "clifford_t"


def test_logical_counts_from_stats_pbc_uses_post_optimization_counts():
    counts = azure_qre.logical_counts_from_stats(
        STATS, "adder_10q_gs_prec5", counts_source="pbc"
    )
    assert (counts.t_count, counts.measurement_count) == (32, 10)
    assert counts.counts_source == "pbc"


def test_logical_counts_from_stats_pbc_falls_back_to_legacy_key():
    legacy = {"num_qubits": 4, "pbc_t_operators": 1274, "pbc_measurement_operators": 4}
    counts = azure_qre.logical_counts_from_stats(legacy, "hhl_4q", counts_source="pbc")
    assert counts.t_count == 1274


def test_logical_counts_from_stats_pbc_defaults_measurements_to_width():
    stats = {"num_qubits": 6, "pbc_rotation_operators": 10}
    counts = azure_qre.logical_counts_from_stats(stats, "x", counts_source="pbc")
    assert counts.measurement_count == 6


def test_logical_counts_from_stats_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unknown counts_source"):
        azure_qre.logical_counts_from_stats(STATS, "x", counts_source="nonsense")


def test_logical_counts_from_stats_rejects_missing_key():
    with pytest.raises(KeyError, match="total_t_family_count"):
        azure_qre.logical_counts_from_stats({"num_qubits": 4}, "x")


def test_read_stats_json_derives_label_from_filename(tmp_path: Path):
    path = tmp_path / "adder_10q_gs_prec5_stats.json"
    path.write_text(json.dumps(STATS), encoding="utf-8")
    counts = azure_qre.read_stats_json(str(path))
    assert counts.label == "adder_10q_gs_prec5"
    assert counts.t_count == 56


# --- Hardware models ---------------------------------------------------------


def test_job_params_omit_qec_scheme_when_unset():
    model = azure_qre.HARDWARE_MODELS["superconducting 1e-3"]
    params = model.job_params()
    assert params == {
        "qubitParams": {"name": "qubit_gate_ns_e3"},
        "errorBudget": 0.01,
    }


def test_job_params_include_qec_scheme_for_majorana_models():
    params = azure_qre.HARDWARE_MODELS["majorana 1e-6"].job_params()
    assert params["qecScheme"] == {"name": "floquet_code"}
    assert params["qubitParams"] == {"name": "qubit_maj_ns_e6"}


def test_resolve_hardware_models_defaults_and_budget_override():
    assert [m.name for m in azure_qre.resolve_hardware_models()] == list(
        azure_qre.DEFAULT_MODEL_NAMES
    )
    models = azure_qre.resolve_hardware_models(["trapped-ion 1e-3"], error_budget=0.001)
    assert [(m.name, m.error_budget) for m in models] == [("trapped-ion 1e-3", 0.001)]


def test_resolve_hardware_models_rejects_unknown_name():
    with pytest.raises(KeyError, match="Unknown hardware model"):
        azure_qre.resolve_hardware_models(["quantum-vacuum-tube"])


def test_to_job_input_omits_zero_ccz_count():
    counts = azure_qre.LogicalResourceCounts("x-4q", 4, 20, 4)
    assert counts.to_job_input() == {
        "numQubits": 4,
        "tCount": 20,
        "measurementCount": 4,
    }
    with_ccz = azure_qre.LogicalResourceCounts("x-4q", 4, 20, 4, ccz_count=3)
    assert with_ccz.to_job_input()["cczCount"] == 3


# --- Estimation against a stubbed estimator ----------------------------------


def _fake_result(physical_qubits: int = 44060) -> dict:
    return {
        "physicalCounts": {
            "physicalQubits": physical_qubits,
            "runtime": 237600,  # nanoseconds
            "breakdown": {
                "logicalDepth": 400,
                "algorithmicLogicalQubits": 21,
                "numTstates": 56,
                "numTfactories": 14,
                "physicalQubitsForAlgorithm": 3402,
                "physicalQubitsForTfactories": 40658,
            },
        },
        "logicalQubit": {"codeDistance": 9},
    }


def _install_fake_qre(monkeypatch, result=None, raises: Exception | None = None):
    """Inject a stand-in for `qdk.estimator` so tests don't need the real one."""
    calls: list[tuple[dict, dict]] = []

    class _FakeLogicalCounts:
        def __init__(self, job_input):
            self.job_input = job_input

        def estimate(self, params):
            calls.append((self.job_input, params))
            if raises is not None:
                raise raises
            return result if result is not None else _fake_result()

    estimator_module = types.SimpleNamespace(LogicalCounts=_FakeLogicalCounts)
    qdk_module = types.SimpleNamespace(estimator=estimator_module)
    monkeypatch.setitem(sys.modules, "qdk", qdk_module)
    monkeypatch.setitem(sys.modules, "qdk.estimator", estimator_module)
    return calls


def test_estimate_circuit_maps_qre_result_fields(monkeypatch):
    calls = _install_fake_qre(monkeypatch)
    counts = azure_qre.LogicalResourceCounts("adder-10q-gs-5", 10, 56, 10)

    result = azure_qre.estimate_circuit(counts)

    assert not result.errors
    assert sorted(result.estimates) == sorted(azure_qre.DEFAULT_MODEL_NAMES)
    estimate = result.estimates["superconducting 1e-3"]
    assert estimate.physical_qubits == 44060
    assert estimate.runtime_seconds == pytest.approx(237600 / 1e9)
    assert estimate.code_distance == 9
    assert estimate.num_t_factories == 14

    # One call per model, each carrying the same logical counts.
    assert len(calls) == 2
    assert calls[0][0] == {"numQubits": 10, "tCount": 56, "measurementCount": 10}
    assert calls[0][1]["qubitParams"] == {"name": "qubit_gate_ns_e3"}
    assert calls[1][1]["qubitParams"] == {"name": "qubit_gate_ns_e4"}


def test_estimate_circuit_records_failures_per_model(monkeypatch):
    _install_fake_qre(monkeypatch, raises=RuntimeError("infeasible error budget"))
    counts = azure_qre.LogicalResourceCounts("adder-10q-gs-5", 10, 56, 10)

    result = azure_qre.estimate_circuit(counts)

    assert not result.estimates
    assert set(result.errors) == set(azure_qre.DEFAULT_MODEL_NAMES)
    assert "infeasible error budget" in result.errors["superconducting 1e-3"]
    assert result.to_dict()["models"]["superconducting 1e-3"]["error"].startswith(
        "RuntimeError"
    )


def test_estimate_circuit_reports_zero_t_without_calling_qre(monkeypatch):
    calls = _install_fake_qre(monkeypatch)
    counts = azure_qre.LogicalResourceCounts("clifford-only-8q", 8, 0, 8)

    result = azure_qre.estimate_circuit(counts)

    assert calls == []
    assert not result.estimates
    assert "requires tCount > 0" in result.errors["superconducting 1e-3"]


def test_estimate_circuits_serializes_to_json_shape(monkeypatch):
    _install_fake_qre(monkeypatch)
    counts = [
        azure_qre.LogicalResourceCounts("adder-10q-gs-5", 10, 56, 10),
        azure_qre.LogicalResourceCounts("qft-4q-sk-1", 4, 12, 4),
    ]

    results = azure_qre.estimate_circuits(
        counts, models=[azure_qre.HARDWARE_MODELS["superconducting 1e-3"]]
    )

    payload = [r.to_dict() for r in results]
    assert json.loads(json.dumps(payload)) == payload  # round-trips cleanly
    assert payload[0]["label"] == "adder-10q-gs-5"
    assert payload[0]["counts_source"] == "clifford_t"
    assert set(payload[0]["models"]) == {"superconducting 1e-3"}
    assert payload[0]["models"]["superconducting 1e-3"]["physical_qubits"] == 44060


def test_import_error_carries_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "qdk", None)
    monkeypatch.setitem(sys.modules, "qdk.estimator", None)
    monkeypatch.setitem(sys.modules, "qsharp", None)
    monkeypatch.setitem(sys.modules, "qsharp.estimator", None)
    assert azure_qre.is_qre_available() is False
    with pytest.raises(ImportError, match="uv sync --extra qre"):
        azure_qre.estimate_circuit(
            azure_qre.LogicalResourceCounts("adder-10q-gs-5", 10, 56, 10)
        )


# --- Estimation against the real Azure QRE (optional dependency) -------------


@pytest.mark.skipif(
    not azure_qre.is_qre_available(),
    reason="optional Azure QRE dependency (qdk) not installed",
)
def test_real_estimator_produces_consistent_physical_costs():
    counts = azure_qre.LogicalResourceCounts("adder-10q-gs-5", 10, 56, 10)
    result = azure_qre.estimate_circuit(counts)

    assert not result.errors, result.errors
    for name in azure_qre.DEFAULT_MODEL_NAMES:
        estimate = result.estimates[name]
        # Error correction always costs more physical qubits than logical ones.
        assert estimate.physical_qubits > counts.num_qubits
        assert estimate.algorithmic_logical_qubits >= counts.num_qubits
        assert estimate.runtime_seconds > 0
        assert estimate.code_distance >= 1
        assert estimate.num_t_states >= counts.t_count
        assert (
            estimate.physical_qubits_for_algorithm
            + estimate.physical_qubits_for_t_factories
            == estimate.physical_qubits
        )


@pytest.mark.skipif(
    not azure_qre.is_qre_available(),
    reason="optional Azure QRE dependency (qdk) not installed",
)
def test_real_estimator_better_qubits_need_smaller_code_distance():
    counts = azure_qre.LogicalResourceCounts("adder-10q-gs-5", 10, 56, 10)
    result = azure_qre.estimate_circuit(counts)
    noisy = result.estimates["superconducting 1e-3"]
    clean = result.estimates["superconducting 1e-4"]
    assert clean.code_distance < noisy.code_distance
    assert clean.physical_qubits < noisy.physical_qubits
