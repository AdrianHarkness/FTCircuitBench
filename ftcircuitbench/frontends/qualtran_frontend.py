"""Qualtran -> FTCircuitBench frontend.

Qualtran describes algorithms as `Bloq`s. A Bloq lowers to a `cirq.Circuit`, and
from there :mod:`ftcircuitbench.frontends.cirq_frontend` carries it into
OpenQASM 2 and the FTCircuitBench pipeline.

`qualtran` is an optional dependency:

    uv sync --extra qualtran      # or: pip install "ftcircuitbench[qualtran]"

**Measurement-based uncomputation.** Qualtran's `And` adjoint is implemented as
measure-and-fix-up rather than as a unitary: it costs zero T gates, which is why
fault-tolerant compilers prefer it. That makes the decomposed circuit
non-unitary, so FTCircuitBench's Clifford+T -> PBC pipeline cannot model it.
Passing `unitary_uncompute=True` substitutes the unitary adjoint of `And` --
the compute circuit reversed and inverted, which is exactly `And†` as a matrix
(:func:`unitary_uncompute_interceptor` asserts nothing less). This makes the
circuit analysable **at a cost**: each substituted uncomputation adds 4 T gates
that the measurement-based version avoids. Resource numbers produced this way
are an upper bound on the measurement-based implementation, and the difference
is exactly `4 x (number of And adjoints)`.

Why 4 and not a Toffoli's 7: `And` is Gidney's *temporary AND*
(arXiv:1709.06648), whose target qubit is guaranteed to start in |0>. Its
compute circuit costs 4 T under that guarantee, and the substituted adjoint is
that circuit inverted. A general ancilla-free Toffoli, which must work on an
arbitrary target state, costs 7 T -- `And` is not a Toffoli, and its unitary
agrees with Toffoli's only on the target-in-|0> columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from qiskit import QuantumCircuit

from .cirq_frontend import cirq_to_qasm2, cirq_to_qiskit, decompose_for_qasm

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import cirq
    from qualtran import Bloq

__all__ = [
    "QUALTRAN_INSTALL_HINT",
    "bloq_to_cirq",
    "bloq_to_qasm2",
    "bloq_to_qiskit",
    "bloq_t_complexity",
    "count_measurement_uncompute",
    "is_qualtran_available",
    "unitary_uncompute_interceptor",
]

QUALTRAN_INSTALL_HINT = (
    "Qualtran interop requires the optional `qualtran` dependency. Install it "
    'with `uv sync --extra qualtran` or `pip install "ftcircuitbench[qualtran]"`.'
)


def is_qualtran_available() -> bool:
    """True when the optional `qualtran` dependency can be imported."""
    try:
        import qualtran  # noqa: F401

        return True
    except ImportError:
        return False


def _require_qualtran() -> Any:
    try:
        import qualtran

        return qualtran
    except ImportError as exc:
        raise ImportError(QUALTRAN_INSTALL_HINT) from exc


def _and_adjoint_bloq(op: Any) -> Optional[Any]:
    """Return the `And` bloq behind `op` if it is a measurement-based adjoint."""
    from qualtran.bloqs.mcmt import And

    gate = op.gate
    if gate is None:
        return None
    # Qualtran bloqs are Cirq gates in current releases; older ones wrap them in
    # BloqAsCirqGate, which exposes the bloq through `.bloq`.
    bloq = getattr(gate, "bloq", gate)
    if isinstance(bloq, And) and bloq.uncompute:
        return bloq
    return None


def unitary_uncompute_interceptor(op: Any) -> Any:
    """Rewrite Qualtran's measurement-based `And†` as its unitary equivalent.

    Returns the `And` compute circuit reversed and inverted, which is `And†`
    exactly. Any other operation returns `NotImplemented`, which tells Cirq to
    fall through to its own decomposition.

    Intended as the `intercepting_decomposer` argument of
    :func:`~ftcircuitbench.frontends.cirq_frontend.decompose_for_qasm`.
    """
    import cirq
    from qualtran.bloqs.mcmt import And

    if _and_adjoint_bloq(op) is None:
        return NotImplemented
    compute = cirq.decompose_once(And().on(*op.qubits))
    return cirq.inverse(compute)


def count_measurement_uncompute(bloq_or_circuit: Any) -> int:
    """Count the measurement-based `And†` operations a Bloq or circuit contains.

    Multiply by 4 for the T gates `unitary_uncompute=True` adds relative to the
    measurement-based implementation. Checking that identity is the point of
    this function: for a Bloq `b`,

        t_measured == bloq_t_complexity(b)["t"] + 4 * count_measurement_uncompute(b)

    where `t_measured` is the T-family count FTCircuitBench reports after
    `bloq_to_qiskit(b, unitary_uncompute=True)`.

    `And†` operations appear only partway down the decomposition -- they are
    inside the top-level Bloq and gone once decomposition finishes -- so this
    decomposes with a stopping rule that holds them in place.
    """
    import cirq

    _require_qualtran()
    from .cirq_frontend import has_qasm_representation

    if isinstance(bloq_or_circuit, cirq.AbstractCircuit):
        circuit: Any = bloq_or_circuit
    else:
        circuit = cirq.Circuit(bloq_or_circuit.as_composite_bloq().to_cirq_circuit())

    def keep(op: Any) -> bool:
        return _and_adjoint_bloq(op) is not None or has_qasm_representation(op)

    ops = cirq.decompose(circuit, keep=keep, on_stuck_raise=None)
    return sum(1 for op in ops if _and_adjoint_bloq(op) is not None)


def bloq_to_cirq(
    bloq: "Bloq",
    unitary_uncompute: bool = False,
    decompose: bool = True,
) -> "cirq.Circuit":
    """Lower a Qualtran Bloq to a Cirq circuit.

    Args:
        bloq: The Bloq to lower.
        unitary_uncompute: Substitute the unitary `And†` for Qualtran's
            measurement-based one. Required for most arithmetic and data-loading
            bloqs to survive FTCircuitBench's unitary-circuit check; adds 4 T
            gates per substitution. See the module docstring.
        decompose: Decompose to QASM-renderable operations. When False, the
            circuit is returned as Qualtran built it (typically a single
            composite operation).

    Raises:
        ImportError: If `qualtran` is not installed.
    """
    import cirq

    _require_qualtran()
    circuit = cirq.Circuit(bloq.as_composite_bloq().to_cirq_circuit())
    if not decompose:
        return circuit
    interceptor = unitary_uncompute_interceptor if unitary_uncompute else None
    return decompose_for_qasm(circuit, intercepting_decomposer=interceptor)


def bloq_to_qasm2(
    bloq: "Bloq",
    unitary_uncompute: bool = False,
    allow_non_unitary: bool = False,
) -> str:
    """Lower a Qualtran Bloq to an OpenQASM 2 string. See :func:`bloq_to_cirq`."""
    circuit = bloq_to_cirq(bloq, unitary_uncompute=unitary_uncompute)
    return cirq_to_qasm2(circuit, decompose=False, allow_non_unitary=allow_non_unitary)


def bloq_to_qiskit(
    bloq: "Bloq",
    unitary_uncompute: bool = False,
    allow_non_unitary: bool = False,
) -> QuantumCircuit:
    """Lower a Qualtran Bloq to a Qiskit circuit, ready for `run_pipeline`."""
    circuit = bloq_to_cirq(bloq, unitary_uncompute=unitary_uncompute)
    return cirq_to_qiskit(circuit, decompose=False, allow_non_unitary=allow_non_unitary)


def bloq_t_complexity(bloq: "Bloq") -> Dict[str, Any]:
    """Qualtran's own analytic cost model for a Bloq.

    Useful as an independent cross-check on what FTCircuitBench measures after
    synthesis: Qualtran counts T gates symbolically from the Bloq's call graph,
    FTCircuitBench counts them in the compiled Clifford+T circuit. The two
    should agree in magnitude, and diverge exactly where synthesis of rotations
    (or a `unitary_uncompute` substitution) adds cost the analytic model does
    not carry.
    """
    _require_qualtran()
    from qualtran.cirq_interop.t_complexity_protocol import t_complexity

    complexity = t_complexity(bloq)
    return {
        "t": complexity.t,
        "clifford": complexity.clifford,
        "rotations": complexity.rotations,
    }
