from __future__ import annotations

import numpy as np
import pytest

from ftcircuitbench.frontends import cirq_frontend, qualtran_frontend

cirq = pytest.importorskip("cirq", reason="optional `cirq` extra not installed")
pytest.importorskip("qualtran", reason="optional `qualtran` extra not installed")

pytestmark = pytest.mark.skipif(
    not qualtran_frontend.is_qualtran_available(),
    reason="optional `qualtran` extra not installed",
)


@pytest.fixture
def qrom_bloq():
    """A small QROAM select-swap block, the notebook demo's building block."""
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM

    data = np.arange(8, dtype=int) * 3 % 8
    return SelectSwapQROM.build_from_data(
        data, target_bitsizes=(3,), log_block_sizes=(1,)
    )


@pytest.fixture
def qft_bloq():
    from qualtran.bloqs.qft import QFTTextBook

    return QFTTextBook(3)


# --- The And-adjoint substitution --------------------------------------------


def test_unitary_uncompute_substitution_is_exactly_the_and_adjoint():
    """The substituted circuit must equal And† as a matrix, not merely resemble it.

    This is the one place the frontend rewrites a circuit rather than lowering
    it, so the rewrite is checked numerically.
    """
    from qualtran.bloqs.mcmt import And

    qubits = cirq.LineQubit.range(3)
    adjoint_op = And(uncompute=True).on(*qubits)
    replacement = qualtran_frontend.unitary_uncompute_interceptor(adjoint_op)

    rebuilt = cirq.Circuit(replacement).unitary(qubit_order=qubits)
    expected = cirq.unitary(And()).conj().T
    assert np.allclose(rebuilt, expected)


def test_unitary_uncompute_substitution_costs_four_t_gates():
    """The documented '+4 T per substitution' figure."""
    from qualtran.bloqs.mcmt import And

    qubits = cirq.LineQubit.range(3)
    replacement = list(
        qualtran_frontend.unitary_uncompute_interceptor(And(uncompute=True).on(*qubits))
    )
    t_like = [
        op
        for op in replacement
        if isinstance(op.gate, cirq.ZPowGate) and abs(op.gate.exponent) == 0.25
    ]
    assert len(t_like) == 4


def test_interceptor_passes_through_unrelated_operations():
    q = cirq.LineQubit.range(2)
    assert (
        qualtran_frontend.unitary_uncompute_interceptor(cirq.H(q[0])) is NotImplemented
    )
    assert (
        qualtran_frontend.unitary_uncompute_interceptor(cirq.CNOT(q[0], q[1]))
        is NotImplemented
    )


def test_compute_and_is_left_alone():
    """Only the *adjoint* is measurement-based; the compute direction is unitary."""
    from qualtran.bloqs.mcmt import And

    q = cirq.LineQubit.range(3)
    assert (
        qualtran_frontend.unitary_uncompute_interceptor(And().on(*q)) is NotImplemented
    )


# --- Lowering ----------------------------------------------------------------


def test_bloq_to_qiskit_refuses_measurement_based_uncompute(qrom_bloq):
    with pytest.raises(cirq_frontend.NonUnitaryCircuitError) as excinfo:
        qualtran_frontend.bloq_to_qiskit(qrom_bloq)
    assert "unitary_uncompute=True" in str(excinfo.value)


def test_bloq_to_qiskit_with_unitary_uncompute(qrom_bloq):
    qc = qualtran_frontend.bloq_to_qiskit(qrom_bloq, unitary_uncompute=True)
    counts = qc.count_ops()
    assert qc.num_qubits > 0
    assert counts.get("t", 0) + counts.get("tdg", 0) > 0
    # No non-unitary instruction survived.
    assert not {"measure", "reset", "if_else"} & set(counts)


def test_bloq_to_cirq_undecomposed_is_a_single_operation(qrom_bloq):
    circuit = qualtran_frontend.bloq_to_cirq(qrom_bloq, decompose=False)
    assert sum(cirq_frontend.op_census(circuit).values()) == 1


def test_bloq_to_qasm2_round_trips_through_qiskit(qft_bloq):
    from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS, loads

    qasm = qualtran_frontend.bloq_to_qasm2(qft_bloq, unitary_uncompute=True)
    qc = loads(qasm, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    assert qc.num_qubits > 0
    assert len(qc.data) > 0


# --- Cross-check against Qualtran's own cost model ----------------------------


def test_t_count_matches_qualtran_plus_substitution_cost(qrom_bloq):
    """FTCircuitBench's measured T count must equal Qualtran's analytic count
    plus exactly 4 T per And-adjoint the frontend made unitary."""
    analytic = qualtran_frontend.bloq_t_complexity(qrom_bloq)["t"]
    substitutions = qualtran_frontend.count_measurement_uncompute(qrom_bloq)
    qc = qualtran_frontend.bloq_to_qiskit(qrom_bloq, unitary_uncompute=True)
    counts = qc.count_ops()
    measured = counts.get("t", 0) + counts.get("tdg", 0)

    assert substitutions > 0, "fixture should exercise measurement-based uncompute"
    assert measured == analytic + 4 * substitutions


def test_bloq_t_complexity_reports_the_expected_fields(qft_bloq):
    complexity = qualtran_frontend.bloq_t_complexity(qft_bloq)
    assert set(complexity) == {"t", "clifford", "rotations"}
    assert complexity["t"] >= 0


def test_count_measurement_uncompute_accepts_a_circuit_too(qrom_bloq):
    from_bloq = qualtran_frontend.count_measurement_uncompute(qrom_bloq)
    circuit = qualtran_frontend.bloq_to_cirq(qrom_bloq, decompose=False)
    from_circuit = qualtran_frontend.count_measurement_uncompute(circuit)
    assert from_bloq == from_circuit > 0


# --- The space/time tradeoff the demo notebook plots --------------------------


@pytest.mark.parametrize("log_block_size", [0, 1, 2])
def test_qroam_block_size_trades_qubits_for_t_gates(log_block_size):
    """Larger QROAM blocks buy T gates with ancilla qubits; every setting must
    survive the whole lowering path."""
    from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM

    data = np.arange(16, dtype=int) % 13
    bloq = SelectSwapQROM.build_from_data(
        data, target_bitsizes=(4,), log_block_sizes=(log_block_size,)
    )
    qc = qualtran_frontend.bloq_to_qiskit(bloq, unitary_uncompute=True)
    counts = qc.count_ops()
    assert qc.num_qubits > 0
    assert counts.get("t", 0) + counts.get("tdg", 0) > 0
