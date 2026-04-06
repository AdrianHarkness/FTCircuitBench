from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

import ftcircuitbench.fidelity as fidelity_mod
from ftcircuitbench.fidelity import calculate_circuit_fidelity, rz_product_fidelity


def test_calculate_fidelity_unitary_path_success(simple_two_qubit_circuit) -> None:
    result = calculate_circuit_fidelity(
        simple_two_qubit_circuit,
        simple_two_qubit_circuit.copy(),
        gridsynth_precision=3,
    )
    assert result["method"] == "unitary_based"
    assert result["status"] == "success"
    assert result["fidelity"] == pytest.approx(1.0, abs=1e-12)


def test_calculate_fidelity_unitary_path_error_status() -> None:
    original = QuantumCircuit(1)
    original.h(0)
    decomposed = QuantumCircuit(2)
    decomposed.cx(0, 1)

    result = calculate_circuit_fidelity(original, decomposed, gridsynth_precision=3)
    assert result["method"] == "unitary_based"
    assert str(result["status"]).startswith("error:")
    assert result["fidelity"] is None


def test_calculate_fidelity_large_without_intermediate_returns_na(large_rz_circuit) -> None:
    result = calculate_circuit_fidelity(
        large_rz_circuit,
        large_rz_circuit.copy(),
        gridsynth_precision=3,
        sk_recursion_degree=1,
        intermediate_qc=None,
    )
    assert result["fidelity"] == "N/A"
    assert result["status"] == "not_available_no_intermediate_circuit"
    assert result["method"] == "rz_product_fidelity_sk"


def test_calculate_fidelity_large_uses_sk_product(monkeypatch, large_rz_circuit) -> None:
    monkeypatch.setattr(
        fidelity_mod,
        "rz_product_fidelity_sk",
        lambda _c, _r: {
            "overall_fidelity": 0.91,
            "individual_fidelities": [0.91],
            "rz_gate_count": 1,
            "status": "success",
        },
    )
    result = calculate_circuit_fidelity(
        large_rz_circuit,
        large_rz_circuit.copy(),
        gridsynth_precision=3,
        sk_recursion_degree=2,
        intermediate_qc=large_rz_circuit,
    )
    assert result["method"] == "rz_product_fidelity_sk"
    assert result["fidelity"] == pytest.approx(0.91)
    assert result["rz_gate_count"] == 1


def test_calculate_fidelity_large_uses_gridsynth_product(monkeypatch, large_rz_circuit) -> None:
    monkeypatch.setattr(
        fidelity_mod,
        "rz_product_fidelity",
        lambda _c, _p: {
            "overall_fidelity": 0.88,
            "individual_fidelities": [0.95, 0.93],
            "rz_gate_count": 2,
            "status": "partial_failure",
        },
    )
    result = calculate_circuit_fidelity(
        large_rz_circuit,
        large_rz_circuit.copy(),
        gridsynth_precision=3,
        intermediate_qc=large_rz_circuit,
    )
    assert result["method"] == "rz_product_fidelity"
    assert result["fidelity"] == pytest.approx(0.88)
    assert result["status"] == "partial_failure"


def test_rz_product_fidelity_no_rz_gates_returns_contract() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["status"] == "no_rz_gates"
    assert result["overall_fidelity"] == "N/A"
    assert result["rz_gate_count"] == 0


def test_rz_product_fidelity_skips_parameterized_rz(monkeypatch) -> None:
    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", lambda *_args, **_kwargs: "T")
    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    qc.rz(theta, 0)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["status"] == "no_rz_gates"
    assert result["rz_gate_count"] == 0


def test_rz_product_fidelity_sequential_success(monkeypatch) -> None:
    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", lambda *_args, **_kwargs: "T")
    qc = QuantumCircuit(1)
    qc.rz(0.2, 0)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["status"] in {"success", "partial_failure"}
    assert result["rz_gate_count"] == 1
    assert isinstance(result["overall_fidelity"], float)
    assert result["multiprocessing_used"] is False


def test_rz_product_fidelity_multiprocessing_fallback(monkeypatch) -> None:
    class _BrokenPool:
        def __enter__(self):
            raise RuntimeError("pool setup failed")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        fidelity_mod.multiprocessing,
        "Pool",
        lambda *args, **kwargs: _BrokenPool(),
    )
    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", lambda *_args, **_kwargs: "W")

    qc = QuantumCircuit(2)
    qc.rz(0.2, 0)
    qc.rz(0.3, 1)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=True)
    assert result["status"] == "success"
    assert result["rz_gate_count"] == 2
    assert result["multiprocessing_used"] is False
