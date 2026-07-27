from __future__ import annotations

import numpy as np
import pytest

from ftcircuitbench.frontends import cirq_frontend

cirq = pytest.importorskip("cirq", reason="optional `cirq` extra not installed")

pytestmark = pytest.mark.skipif(
    not cirq_frontend.is_cirq_available(),
    reason="optional `cirq` extra not installed",
)


@pytest.fixture
def clifford_t_cirq_circuit():
    """A small Clifford+T circuit with no rotations and no measurement."""
    q = cirq.LineQubit.range(3)
    return cirq.Circuit(
        [
            cirq.H(q[0]),
            cirq.T(q[1]),
            cirq.CNOT(q[0], q[2]),
            cirq.S(q[2]),
            cirq.CZ(q[1], q[2]),
            cirq.T(q[0]) ** -1,
        ]
    )


def _phase_normalized(matrix: np.ndarray) -> np.ndarray:
    largest = matrix.flat[int(np.argmax(np.abs(matrix)))]
    return matrix * (abs(largest) / largest)


# --- QASM-representability stopping rule -------------------------------------


def test_has_qasm_representation_accepts_standard_gates():
    q = cirq.LineQubit.range(2)
    assert cirq_frontend.has_qasm_representation(cirq.H(q[0]))
    assert cirq_frontend.has_qasm_representation(cirq.CNOT(q[0], q[1]))
    assert cirq_frontend.has_qasm_representation(cirq.T(q[0]))


def test_has_qasm_representation_rejects_a_gate_without_qasm():
    class NoQasmGate(cirq.Gate):
        def _num_qubits_(self):
            return 1

        def _decompose_(self, qubits):
            yield cirq.H(qubits[0])
            yield cirq.T(qubits[0])

        def _qasm_(self, args, qubits):
            return NotImplemented

    q = cirq.LineQubit(0)
    assert not cirq_frontend.has_qasm_representation(NoQasmGate().on(q))


def test_decompose_for_qasm_expands_only_what_qasm_needs():
    class NoQasmGate(cirq.Gate):
        def _num_qubits_(self):
            return 1

        def _decompose_(self, qubits):
            yield cirq.H(qubits[0])
            yield cirq.T(qubits[0])

        def _qasm_(self, args, qubits):
            return NotImplemented

    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([NoQasmGate().on(q[0]), cirq.CNOT(q[0], q[1])])
    out = cirq_frontend.decompose_for_qasm(circuit)
    census = cirq_frontend.op_census(out)
    # The custom gate expanded; the CNOT, which QASM can express, did not.
    assert "CXPowGate" in census
    assert "NoQasmGate" not in census
    assert census.get("HPowGate") == 1


def test_decompose_for_qasm_preserves_native_clifford_t_structure(
    clifford_t_cirq_circuit,
):
    """The stopping rule must not rewrite H/CNOT into rotations.

    `cirq.decompose` with no stopping rule lowers to a hardware gateset and
    turns this circuit into X/Y/Z power gates; that would hand FTCircuitBench
    rotations to re-synthesise that it never needed to see.
    """
    kept = cirq_frontend.op_census(
        cirq_frontend.decompose_for_qasm(clifford_t_cirq_circuit)
    )
    full = cirq_frontend.op_census(
        cirq.Circuit(cirq.decompose(clifford_t_cirq_circuit))
    )

    assert "YPowGate" not in kept
    assert "YPowGate" in full
    assert sum(kept.values()) < sum(full.values())


def test_op_census_counts_by_gate_type(clifford_t_cirq_circuit):
    census = cirq_frontend.op_census(clifford_t_cirq_circuit)
    assert census["ZPowGate"] == 3  # T, S, T**-1
    assert census["CXPowGate"] == 1
    assert census["CZPowGate"] == 1


# --- Conversion --------------------------------------------------------------


def test_cirq_to_qasm2_parses_back_into_qiskit(clifford_t_cirq_circuit):
    qasm = cirq_frontend.cirq_to_qasm2(clifford_t_cirq_circuit)
    assert "OPENQASM 2.0;" in qasm  # Cirq prefixes a generator comment
    qc = cirq_frontend.cirq_to_qiskit(clifford_t_cirq_circuit)
    assert qc.num_qubits == 3
    assert dict(qc.count_ops()) == {"h": 1, "t": 1, "tdg": 1, "cx": 1, "s": 1, "cz": 1}


def test_cirq_to_qiskit_preserves_the_unitary_up_to_bit_order(
    clifford_t_cirq_circuit,
):
    """Cirq is big-endian, Qiskit little-endian; the circuits agree after reversal."""
    from qiskit.quantum_info import Operator

    qc = cirq_frontend.cirq_to_qiskit(clifford_t_cirq_circuit)
    from_cirq = cirq.unitary(clifford_t_cirq_circuit)
    from_qiskit = Operator(qc.reverse_bits()).data

    assert np.allclose(_phase_normalized(from_cirq), _phase_normalized(from_qiskit))


# --- Non-unitary handling ----------------------------------------------------


def test_terminal_measurements_are_accepted():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.measure(*q)])
    qasm = cirq_frontend.cirq_to_qasm2(circuit)
    assert "measure" in qasm


def test_mid_circuit_measurement_is_rejected():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        [cirq.H(q[0]), cirq.measure(q[0], key="m"), cirq.CNOT(q[0], q[1])]
    )
    with pytest.raises(cirq_frontend.NonUnitaryCircuitError, match="non-unitary"):
        cirq_frontend.cirq_to_qasm2(circuit)


def test_reset_is_rejected_and_named_in_the_error():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(q[0]), cirq.reset(q[0]), cirq.CNOT(q[0], q[1])])
    with pytest.raises(cirq_frontend.NonUnitaryCircuitError, match="ResetChannel"):
        cirq_frontend.cirq_to_qasm2(circuit)


def test_allow_non_unitary_exports_anyway():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(q[0]), cirq.reset(q[0]), cirq.CNOT(q[0], q[1])])
    qasm = cirq_frontend.cirq_to_qasm2(circuit, allow_non_unitary=True)
    assert "reset" in qasm


def test_non_unitary_error_points_at_the_escape_hatches():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(q[0]), cirq.reset(q[0]), cirq.CNOT(q[0], q[1])])
    with pytest.raises(cirq_frontend.NonUnitaryCircuitError) as excinfo:
        cirq_frontend.cirq_to_qasm2(circuit)
    message = str(excinfo.value)
    assert "unitary_uncompute=True" in message
    assert "allow_non_unitary=True" in message
