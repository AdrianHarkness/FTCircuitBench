from __future__ import annotations

from qiskit import QuantumCircuit

from ftcircuitbench.transpilers.sk_transpiler import (
    transpile_to_solovay_kitaev_clifford_t,
)


def test_sk_no_synthesis_needed_does_not_crash() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.s(1)
    qc.cx(0, 1)

    out = transpile_to_solovay_kitaev_clifford_t(qc, recursion_degree=1)
    assert isinstance(out, QuantumCircuit)
    assert out.num_qubits == qc.num_qubits
    assert "rz" not in out.count_ops()


def test_sk_return_intermediate_returns_tuple() -> None:
    qc = QuantumCircuit(1)
    qc.rz(0.2, 0)

    inter, out = transpile_to_solovay_kitaev_clifford_t(
        qc, recursion_degree=1, return_intermediate=True
    )
    assert isinstance(inter, QuantumCircuit)
    assert isinstance(out, QuantumCircuit)
    assert "rz" in inter.count_ops()
    assert "rz" not in out.count_ops()


def test_sk_removes_final_measurements_by_default() -> None:
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    out = transpile_to_solovay_kitaev_clifford_t(qc, recursion_degree=1)
    assert "measure" not in out.count_ops()
