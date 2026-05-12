"""Snapshot regression tests for the layering pipeline.

These tests pin the exact observable output (layer count, layer sizes, the per-layer
Pauli sequence, and `optimize_t` convergence trace) of the current implementation on
two fixed Clifford+T circuits. Any refactor that changes these — even one that
preserves the structural invariants — will trip this test, forcing an explicit
review of the change.

When a refactor *intentionally* changes the algorithm (e.g., a smarter layer_v2),
update the SNAPSHOT_* constants below and document the rationale in the commit
message.
"""

from __future__ import annotations

import pytest
from _pbc_helpers import paulis_with_signs_in_layer

from ftcircuitbench.pbc_converter.r_pauli_circ import RotationPauliCirc

# ---------- Snapshot constants captured 2026-05-08 ----------
# Source circuits live in conftest.py:
#   - small_clifford_t_circuit (3 qubits, 5 T-family gates)
#   - random_clifford_t_circuit (4 qubits, seed 20260508, 11 T-family gates)


# small_clifford_t_circuit: every rotation reduces to a Pauli string that pairwise
# commutes with the others, so both bare and v2 collapse them into one layer.
SMALL_LAYERS_BARE = [
    [("XII", False), ("XZI", False), ("IIX", True), ("XII", False), ("XZX", False)],
]
SMALL_LAYERS_V2 = SMALL_LAYERS_BARE
SMALL_LAYERS_SINGLETON = [
    [("XII", False)],
    [("XZI", False)],
    [("IIX", True)],
    [("XII", False)],
    [("XZX", False)],
]

# random_clifford_t_circuit (seed 20260508): 11 rotations
RANDOM_BARE_SIZES = [5, 6]
RANDOM_V2_SIZES = [9, 2]
RANDOM_SINGLETON_SIZES = [1] * 11

RANDOM_V2_PAULIS = [
    [
        ("IZII", True),
        ("ZIII", True),
        ("ZIII", True),
        ("ZIII", True),
        ("ZIZZ", True),
        ("ZIZZ", False),
        ("IIIZ", True),
        ("ZIZZ", True),
        ("ZIZZ", True),
    ],
    [("XIXZ", False), ("XIXZ", False)],
]

# optimize_t convergence traces for the random circuit
OPTIMIZE_T_TRACE_BARE = [11, 5, 3]
OPTIMIZE_T_TRACE_V2 = [11, 3]


# ---------- Small fixture circuit ----------


def _layer(rpc_circuit, method: str):
    rpc = RotationPauliCirc(rpc_circuit)
    err = rpc.process(ifprint=False)
    assert err is False
    rpc.layering(method=method, ifprint=False)
    return rpc


@pytest.mark.parametrize(
    "method,expected",
    [
        ("bare", SMALL_LAYERS_BARE),
        ("v2", SMALL_LAYERS_V2),
        ("singleton", SMALL_LAYERS_SINGLETON),
    ],
)
def test_small_circuit_layer_snapshot(
    small_clifford_t_circuit, method, expected
) -> None:
    rpc = _layer(small_clifford_t_circuit, method)
    actual = [paulis_with_signs_in_layer(layer) for layer in rpc.t_layers]
    assert actual == expected


# ---------- Random seeded circuit ----------


@pytest.mark.parametrize(
    "method,expected_sizes",
    [
        ("bare", RANDOM_BARE_SIZES),
        ("v2", RANDOM_V2_SIZES),
        ("singleton", RANDOM_SINGLETON_SIZES),
    ],
)
def test_random_circuit_layer_sizes(
    random_clifford_t_circuit, method, expected_sizes
) -> None:
    rpc = _layer(random_clifford_t_circuit, method)
    actual = [layer.stab_counts for layer in rpc.t_layers]
    assert actual == expected_sizes


def test_random_circuit_v2_full_pauli_snapshot(random_clifford_t_circuit) -> None:
    """The strongest snapshot: exact Pauli strings + signs in each v2 layer.

    If this test fails after a refactor, inspect the diff carefully — it indicates
    either (a) a real bug, or (b) an intentional algorithm change that warrants
    a snapshot update with justification.
    """
    rpc = _layer(random_clifford_t_circuit, "v2")
    actual = [paulis_with_signs_in_layer(layer) for layer in rpc.t_layers]
    assert actual == RANDOM_V2_PAULIS


# ---------- optimize_t convergence ----------


@pytest.mark.parametrize(
    "method,expected_trace",
    [
        ("bare", OPTIMIZE_T_TRACE_BARE),
        ("v2", OPTIMIZE_T_TRACE_V2),
    ],
)
def test_optimize_t_convergence_trace(
    random_clifford_t_circuit, method, expected_trace
) -> None:
    """`optimize_t` returns the gate-count trajectory across iterations.

    Locking the trace catches both correctness regressions (wrong final count)
    and convergence regressions (more iterations than before).
    """
    rpc = RotationPauliCirc(random_clifford_t_circuit)
    err = rpc.process(ifprint=False)
    assert err is False
    tracker = rpc.optimize_t(maxiter=10, ifprint=False, layering_method=method)
    assert tracker == expected_trace
