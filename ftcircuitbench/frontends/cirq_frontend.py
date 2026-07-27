"""Cirq -> FTCircuitBench frontend.

Logical-circuit generators in the Cirq ecosystem (Qualtran, pyLIQTR, or
hand-built Cirq) emit `cirq.Circuit` objects. This module lowers one of those
into the OpenQASM 2 that FTCircuitBench's pipeline consumes, which makes
FTCircuitBench the middle layer between logical circuit construction and
physical resource estimation:

    Qualtran / pyLIQTR  ->  cirq.Circuit  ->  OpenQASM 2  ->  FTCircuitBench
      (this module)                                       ->  Azure QRE
                                              (ftcircuitbench.resource_estimation)

`cirq-core` is an optional dependency:

    uv sync --extra cirq          # or: pip install "ftcircuitbench[cirq]"

Two things this module is careful about.

**Decomposition depth.** `cirq.decompose` with no stopping rule runs all the way
to a hardware-style gateset, rewriting `H`/`CNOT` into `Y**0.5`/`CZ` power gates
and inflating the circuit several-fold (QFT(4): 1943 gates at depth 946 instead
of 519 at depth 370). :func:`decompose_for_qasm` instead stops as soon as every
operation has a direct OpenQASM 2 representation, keeping the exported circuit
compact and close to the H/CNOT/T structure the generator emitted. Compiled
resource counts are the same either way — the extra rotations a full
decomposition produces carry Clifford+T angles (multiples of pi/4) that
synthesis reproduces exactly — so the stopping rule buys a faithful, compact
intermediate representation, not a smaller T count.

**Unitarity.** FTCircuitBench's Clifford+T and PBC analysis assumes a unitary
circuit. Cirq circuits from Qualtran routinely contain measurement-based
uncomputation, which is not unitary. Those are reported by default rather than
silently analysed; see `unitary_uncompute` in
:mod:`ftcircuitbench.frontends.qualtran_frontend`.

**Qubit ordering.** Cirq and Qiskit disagree on bit significance, so the
converted circuit satisfies `cirq.unitary(c) == Operator(qc.reverse_bits())`
up to global phase, not `Operator(qc)`. Gate-to-qubit assignment is preserved
exactly, so resource counts are unaffected; only state-vector comparisons need
the reversal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from qiskit import QuantumCircuit

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import cirq

__all__ = [
    "CIRQ_INSTALL_HINT",
    "NonUnitaryCircuitError",
    "cirq_to_qasm2",
    "cirq_to_qiskit",
    "decompose_for_qasm",
    "has_qasm_representation",
    "is_cirq_available",
    "op_census",
]

CIRQ_INSTALL_HINT = (
    "Cirq interop requires the optional `cirq-core` dependency. Install it with "
    '`uv sync --extra cirq` or `pip install "ftcircuitbench[cirq]"`.'
)

# Interceptor signature: given an operation, return replacement operations, or
# NotImplemented to fall through to Cirq's own decomposition.
Interceptor = Callable[[Any], Any]


class NonUnitaryCircuitError(ValueError):
    """Raised when a circuit carries operations FTCircuitBench cannot analyse.

    Measurement, reset, and classically-controlled operations have no place in
    the Clifford+T -> PBC pipeline, which models a unitary circuit. Terminal
    measurements are the one exception: the pipeline strips those itself.
    """


def is_cirq_available() -> bool:
    """True when the optional `cirq-core` dependency can be imported."""
    try:
        import cirq  # noqa: F401

        return True
    except ImportError:
        return False


def _require_cirq() -> Any:
    try:
        import cirq

        return cirq
    except ImportError as exc:
        raise ImportError(CIRQ_INSTALL_HINT) from exc


def has_qasm_representation(op: Any) -> bool:
    """True when Cirq can emit this operation as OpenQASM 2 directly.

    Used as the stopping rule for :func:`decompose_for_qasm`. Cirq's QASM
    protocol needs a qubit-id map to render an operation, so a throwaway one is
    supplied here — only whether rendering succeeds matters, not the result.
    """
    cirq = _require_cirq()
    if cirq.is_measurement(op):
        return True
    args = cirq.QasmArgs(
        qubit_id_map={qubit: f"q[{i}]" for i, qubit in enumerate(op.qubits)}
    )
    try:
        return cirq.qasm(op, args=args, default=None) is not None
    except Exception:
        # A gate whose _qasm_ raises is, for our purposes, not QASM-expressible.
        return False


def decompose_for_qasm(
    circuit: "cirq.AbstractCircuit",
    intercepting_decomposer: Optional[Interceptor] = None,
) -> "cirq.Circuit":
    """Decompose only as far as OpenQASM 2 requires.

    Stops at every operation that already has a QASM representation, so the
    Clifford+T structure a generator emitted survives instead of being rewritten
    into rotations. Operations that can be neither rendered nor decomposed are
    left in place; :func:`cirq_to_qasm2` reports them when the export fails.

    Args:
        circuit: Any Cirq circuit (`Circuit` or `FrozenCircuit`).
        intercepting_decomposer: Optional per-operation rewrite applied before
            Cirq's own decomposition, e.g.
            :func:`~ftcircuitbench.frontends.qualtran_frontend.unitary_uncompute_interceptor`.
    """
    cirq = _require_cirq()
    return cirq.Circuit(
        cirq.decompose(
            circuit,
            keep=has_qasm_representation,
            intercepting_decomposer=intercepting_decomposer,
            on_stuck_raise=None,
        )
    )


def op_census(circuit: "cirq.AbstractCircuit") -> Dict[str, int]:
    """Count operations by gate type — a quick look at what a generator emitted."""
    census: Dict[str, int] = {}
    for op in circuit.all_operations():
        name = type(op.gate).__name__ if op.gate is not None else type(op).__name__
        census[name] = census.get(name, 0) + 1
    return dict(sorted(census.items(), key=lambda kv: (-kv[1], kv[0])))


def _non_unitary_ops(circuit: "cirq.AbstractCircuit") -> List[Any]:
    """Operations that block Clifford+T / PBC analysis.

    Terminal measurements are excluded: `prepare_input` strips those before
    transpilation, so they cost nothing and change nothing.
    """
    cirq = _require_cirq()
    measurements_terminal = circuit.are_all_measurements_terminal()
    offenders = []
    for op in circuit.all_operations():
        if cirq.has_unitary(op):
            continue
        if measurements_terminal and cirq.is_measurement(op):
            continue
        offenders.append(op)
    return offenders


def _describe_offenders(offenders: List[Any]) -> str:
    names: Dict[str, int] = {}
    for op in offenders:
        gate = op.gate
        name = type(gate).__name__ if gate is not None else type(op).__name__
        names[name] = names.get(name, 0) + 1
    return ", ".join(f"{name} x{count}" for name, count in sorted(names.items()))


def cirq_to_qasm2(
    circuit: "cirq.AbstractCircuit",
    decompose: bool = True,
    allow_non_unitary: bool = False,
    intercepting_decomposer: Optional[Interceptor] = None,
) -> str:
    """Convert a Cirq circuit to an OpenQASM 2 string.

    Args:
        circuit: The Cirq circuit to convert.
        decompose: Run :func:`decompose_for_qasm` first. Leave this on unless
            the circuit is already expressed in QASM-renderable gates.
        allow_non_unitary: Emit QASM even when the circuit carries mid-circuit
            measurement, reset, or classical control. Off by default because
            FTCircuitBench's downstream analysis would silently misreport such a
            circuit.
        intercepting_decomposer: See :func:`decompose_for_qasm`.

    Raises:
        NonUnitaryCircuitError: If non-unitary operations survive decomposition
            and `allow_non_unitary` is False.
        ImportError: If `cirq-core` is not installed.
    """
    cirq = _require_cirq()
    prepared = (
        decompose_for_qasm(circuit, intercepting_decomposer=intercepting_decomposer)
        if decompose
        else cirq.Circuit(circuit)
    )

    if not allow_non_unitary:
        offenders = _non_unitary_ops(prepared)
        if offenders:
            raise NonUnitaryCircuitError(
                f"Circuit contains {len(offenders)} non-unitary operation(s) that "
                f"FTCircuitBench's Clifford+T / PBC pipeline cannot model: "
                f"{_describe_offenders(offenders)}. These usually come from "
                "measurement-based uncomputation; pass unitary_uncompute=True to "
                "substitute the unitary equivalent (which changes the T count), or "
                "allow_non_unitary=True to export anyway."
            )

    return cirq.qasm(prepared)


def cirq_to_qiskit(
    circuit: "cirq.AbstractCircuit",
    decompose: bool = True,
    allow_non_unitary: bool = False,
    intercepting_decomposer: Optional[Interceptor] = None,
) -> QuantumCircuit:
    """Convert a Cirq circuit to a Qiskit `QuantumCircuit` via OpenQASM 2.

    The result is ready for `ftcircuitbench.api.run_pipeline`. Arguments match
    :func:`cirq_to_qasm2`.
    """
    from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS, loads

    qasm = cirq_to_qasm2(
        circuit,
        decompose=decompose,
        allow_non_unitary=allow_non_unitary,
        intercepting_decomposer=intercepting_decomposer,
    )
    return loads(qasm, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
