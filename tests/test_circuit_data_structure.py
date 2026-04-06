from __future__ import annotations

from qiskit import QuantumCircuit

from ftcircuitbench import analyze_clifford_t_circuit, analyze_pbc_circuit


def test_quantum_circuit_data_entries_have_expected_fields() -> None:
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)

    assert len(qc.data) == 3
    for entry in qc.data:
        assert hasattr(entry, "operation")
        assert hasattr(entry, "qubits")
        assert hasattr(entry, "clbits")


def test_clifford_t_analyzer_returns_basic_counts() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.t(0)
    qc.tdg(1)
    qc.cx(0, 1)

    stats = analyze_clifford_t_circuit(qc)
    assert stats["t_count"] == 1
    assert stats["tdg_count"] == 1
    assert stats["total_t_family_count"] == 2
    assert stats["total_gate_count"] == len(qc.data)


def test_pbc_analyzer_accepts_minimal_pbc_like_circuit() -> None:
    qc = QuantumCircuit(1)
    qc.rz(0.2, 0)

    stats = analyze_pbc_circuit(qc, pbc_conversion_stats={"pbc_t_operators": 0})
    assert "pbc_t_operators" in stats
