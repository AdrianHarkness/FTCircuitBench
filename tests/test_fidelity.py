from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

import ftcircuitbench.fidelity as fidelity_mod
from ftcircuitbench.fidelity import (
    MAX_QUBITS_FOR_FIDELITY,
    calculate_circuit_fidelity,
    rz_product_fidelity,
)


@pytest.mark.parametrize("n_qubits", [1, 2, 4, MAX_QUBITS_FOR_FIDELITY])
def test_calculate_circuit_fidelity_small_circuits_are_numeric(n_qubits: int) -> None:
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    if n_qubits > 1:
        qc.cx(0, 1)

    result = calculate_circuit_fidelity(qc, qc.copy(), gridsynth_precision=3)
    assert result["method"] == "unitary_based"
    assert isinstance(result["fidelity"], float)
    assert result["fidelity"] == pytest.approx(1.0, abs=1e-12)


def test_rz_product_fidelity_reports_failed_decomposition(monkeypatch) -> None:
    # Force a low-fidelity decomposition by returning an empty/identity-like sequence
    monkeypatch.setattr(
        fidelity_mod, "_run_gridsynth_cli", lambda *_args, **_kwargs: "H"
    )
    qc = QuantumCircuit(1)
    qc.rz(0.6, 0)

    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["rz_gate_count"] == 1
    assert "overall_fidelity" in result
    assert isinstance(result["overall_fidelity"], float)


def test_calculate_circuit_fidelity_large_requires_intermediate() -> None:
    qc = QuantumCircuit(MAX_QUBITS_FOR_FIDELITY + 1)
    qc.rz(0.3, 0)

    result = calculate_circuit_fidelity(
        qc,
        qc.copy(),
        gridsynth_precision=3,
        intermediate_qc=None,
    )
    assert result["fidelity"] == "N/A"
    assert result["status"] == "not_available_no_intermediate_circuit"
