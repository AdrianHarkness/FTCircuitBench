#!/usr/bin/env python3
"""Export a Cirq circuit to OpenQASM 2 for FTCircuitBench, from *any* environment.

This script deliberately imports nothing from `ftcircuitbench`. pyLIQTR releases
pin `numpy<2` and `qualtran==0.4.0`, which cannot coexist with FTCircuitBench's
dependency floor, so a pyLIQTR circuit has to cross the boundary as a file:

    # in the pyLIQTR environment
    python tools/export_cirq_qasm.py my_module:build_circuit -o heisenberg_be.qasm

    # in the FTCircuitBench environment
    uv run python analyze_circuit.py heisenberg_be.qasm --pipeline gs

Circuits from a toolchain that FTCircuitBench *can* host in-process (Cirq,
Qualtran) do not need this script -- use `import_circuit.py`, which shares the
same decomposition rules through `ftcircuitbench.frontends`.

The stopping rule below mirrors `ftcircuitbench.frontends.cirq_frontend`: stop
decomposing as soon as every operation has a direct OpenQASM 2 representation,
so the generator's native Clifford+T structure survives instead of being
rewritten into rotations FTCircuitBench would have to re-synthesise.
"""

import argparse
import importlib
import json
import sys
from typing import Any, List

import cirq


def has_qasm_representation(op: Any) -> bool:
    """True when Cirq can emit this operation as OpenQASM 2 directly."""
    if cirq.is_measurement(op):
        return True
    args = cirq.QasmArgs(
        qubit_id_map={qubit: f"q[{i}]" for i, qubit in enumerate(op.qubits)}
    )
    try:
        return cirq.qasm(op, args=args, default=None) is not None
    except Exception:
        return False


def _and_compute_gate() -> Any:
    """The unitary `And` compute gate, across Qualtran's Bloq/Gate rename.

    Qualtran >=0.5 makes bloqs Cirq gates directly; 0.4 (which pyLIQTR pins)
    wraps them in `BloqAsCirqGate`.
    """
    from qualtran.bloqs.mcmt import And

    compute = And()
    if isinstance(compute, cirq.Gate):
        return compute
    from qualtran.cirq_interop import BloqAsCirqGate

    return BloqAsCirqGate(compute)


def unitary_uncompute_interceptor(op: Any) -> Any:
    """Rewrite Qualtran's measurement-based `And†` as its unitary equivalent.

    Mirrors `ftcircuitbench.frontends.qualtran_frontend`. Adds 4 T gates per
    substitution relative to the measurement-based implementation.
    """
    from qualtran.bloqs.mcmt import And

    gate = op.gate
    if gate is None:
        return NotImplemented
    bloq = getattr(gate, "bloq", gate)
    if not (isinstance(bloq, And) and bloq.uncompute):
        return NotImplemented
    return cirq.inverse(cirq.decompose_once(_and_compute_gate().on(*op.qubits)))


def resolve_target(target: str) -> Any:
    if ":" not in target:
        raise SystemExit(f"Target '{target}' must be in 'module:attribute' form.")
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    # Dotted attributes resolve too, so classmethod factories work as targets.
    obj: Any = module
    for part in attribute.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise SystemExit(
                f"'{module_name}' has no attribute path '{attribute}' "
                f"(failed at '{part}')."
            ) from exc
    return obj


def _parse_json_arg(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def as_circuit(obj: Any) -> cirq.Circuit:
    """Coerce a resolved object into a Cirq circuit."""
    if isinstance(obj, cirq.AbstractCircuit):
        return cirq.Circuit(obj)
    if isinstance(obj, cirq.Gate):
        return cirq.Circuit(obj.on(*cirq.LineQubit.range(cirq.num_qubits(obj))))
    if isinstance(obj, cirq.Operation):
        return cirq.Circuit(obj)
    raise SystemExit(
        f"Target resolved to {type(obj).__name__}; expected a Cirq circuit, gate, "
        "or operation (or a callable returning one)."
    )


def describe(circuit: cirq.Circuit) -> None:
    census: dict = {}
    for op in circuit.all_operations():
        name = type(op.gate).__name__ if op.gate is not None else type(op).__name__
        census[name] = census.get(name, 0) + 1
    ordered = sorted(census.items(), key=lambda kv: (-kv[1], kv[0]))
    print(f"  qubits: {len(circuit.all_qubits())}")
    print(f"  ops:    {sum(census.values())}")
    print(f"  census: {dict(ordered)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a Cirq circuit to OpenQASM 2 for FTCircuitBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        help="Import target as 'module:attribute'. May name a Cirq circuit, "
        "gate, or a callable returning one.",
    )
    parser.add_argument(
        "--args",
        nargs="*",
        default=None,
        help="Positional arguments for the target, each parsed as JSON.",
    )
    parser.add_argument(
        "--kwargs", default=None, help="Keyword arguments as a JSON object."
    )
    parser.add_argument("--output", "-o", required=True, help="Output QASM path.")
    parser.add_argument(
        "--unitary-uncompute",
        action="store_true",
        help="Substitute the unitary And-adjoint for Qualtran's measurement-based "
        "one (pyLIQTR circuits use it heavily). Adds 4 T gates per substitution. "
        "Requires qualtran to be importable.",
    )
    parser.add_argument(
        "--allow-non-unitary",
        action="store_true",
        help="Export even when non-unitary operations survive decomposition. "
        "FTCircuitBench's Clifford+T / PBC analysis cannot model those.",
    )
    args = parser.parse_args()

    obj = resolve_target(args.target)
    call_args: List[Any] = (
        [_parse_json_arg(a) for a in args.args] if args.args is not None else []
    )
    call_kwargs = json.loads(args.kwargs) if args.kwargs else {}
    if call_args or call_kwargs:
        obj = obj(*call_args, **call_kwargs)
    elif callable(obj) and not isinstance(
        obj, (cirq.AbstractCircuit, cirq.Gate, cirq.Operation)
    ):
        obj = obj()

    circuit = as_circuit(obj)
    print("As given:")
    describe(circuit)

    interceptor = unitary_uncompute_interceptor if args.unitary_uncompute else None
    decomposed = cirq.Circuit(
        cirq.decompose(
            circuit,
            keep=has_qasm_representation,
            intercepting_decomposer=interceptor,
            on_stuck_raise=None,
        )
    )
    print("Decomposed for QASM:")
    describe(decomposed)

    terminal = decomposed.are_all_measurements_terminal()
    offenders = [
        op
        for op in decomposed.all_operations()
        if not cirq.has_unitary(op) and not (terminal and cirq.is_measurement(op))
    ]
    if offenders and not args.allow_non_unitary:
        names: dict = {}
        for op in offenders:
            gate = op.gate
            name = type(gate).__name__ if gate is not None else type(op).__name__
            names[name] = names.get(name, 0) + 1
        print(
            f"\nERROR: {len(offenders)} non-unitary operation(s) survive "
            f"decomposition ({names}). FTCircuitBench's Clifford+T / PBC pipeline "
            "models unitary circuits only. Re-run with --allow-non-unitary to "
            "export anyway.",
            file=sys.stderr,
        )
        return 1

    qasm = cirq.qasm(decomposed)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(qasm)
    print(f"\nWrote {args.output} ({len(qasm.splitlines())} lines)")
    print(f"Next (FTCircuitBench env): analyze_circuit.py {args.output} --pipeline gs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
