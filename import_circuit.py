#!/usr/bin/env python3
"""CLI: import a Cirq or Qualtran circuit into FTCircuitBench as OpenQASM 2.

Resolves a `module:attribute` target to a Qualtran Bloq or a Cirq circuit,
lowers it to OpenQASM 2, and optionally runs the FTCircuitBench pipeline on the
result. See `ftcircuitbench.frontends` for the library entry points.

    uv run python import_circuit.py qualtran.bloqs.qft:QFTTextBook --args 4 \\
        --output qasm/qualtran/qft_4q.qasm --unitary-uncompute --analyze
"""

import argparse
import importlib
import json
import os
import sys
from typing import Any, List

from ftcircuitbench.benchmark_utils import print_table
from ftcircuitbench.frontends import (
    NonUnitaryCircuitError,
    bloq_to_cirq,
    cirq_to_qasm2,
    is_cirq_available,
    is_qualtran_available,
    op_census,
)
from ftcircuitbench.frontends.cirq_frontend import CIRQ_INSTALL_HINT


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the circuit importer."""
    parser = argparse.ArgumentParser(
        description="Import a Cirq/Qualtran circuit into FTCircuitBench as OpenQASM 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        help="Import target as 'module:attribute', e.g. "
        "'qualtran.bloqs.qft:QFTTextBook'. The attribute may be a Bloq, a Cirq "
        "circuit, or a callable returning either.",
    )
    parser.add_argument(
        "--args",
        nargs="*",
        default=None,
        help="Positional arguments for the target, each parsed as JSON "
        "(falling back to a plain string).",
    )
    parser.add_argument(
        "--kwargs",
        default=None,
        help="Keyword arguments for the target as a JSON object, e.g. "
        "'{\"num_controls\": 2}'. JSON arrays arrive as lists; targets that "
        "require tuples (many Qualtran constructors do) are easier to reach "
        "through a small factory function you point this CLI at.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path for the OpenQASM 2 file. Defaults to "
        "qasm/imported/<attribute>.qasm.",
    )
    parser.add_argument(
        "--unitary-uncompute",
        action="store_true",
        help="Substitute the unitary And-adjoint for Qualtran's measurement-based "
        "one, so the circuit becomes analysable. Adds 4 T gates per substitution.",
    )
    parser.add_argument(
        "--allow-non-unitary",
        action="store_true",
        help="Export even when mid-circuit measurement, reset, or classical "
        "control survives. FTCircuitBench will misreport such a circuit.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run the FTCircuitBench pipeline on the exported QASM.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["gs", "sk", "both"],
        default="gs",
        help="Pipeline to run when --analyze is given.",
    )
    parser.add_argument(
        "--gridsynth-precision",
        type=int,
        default=5,
        help="Gridsynth precision when --analyze is given.",
    )
    return parser.parse_args()


def _parse_json_arg(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def resolve_target(target: str) -> Any:
    """Resolve a 'module:attribute' string to the object it names."""
    if ":" not in target:
        raise SystemExit(
            f"Target '{target}' must be in 'module:attribute' form, e.g. "
            "'qualtran.bloqs.qft:QFTTextBook'."
        )
    module_name, _, attribute = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"Could not import module '{module_name}': {exc}") from exc
    # Dotted attributes resolve too, so classmethod factories such as
    # 'SelectSwapQROM.build_from_data' work as targets.
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


def build_circuit_object(args: argparse.Namespace) -> Any:
    """Resolve the target and call it if arguments were supplied."""
    obj = resolve_target(args.target)
    call_args: List[Any] = (
        [_parse_json_arg(a) for a in args.args] if args.args is not None else []
    )
    call_kwargs = json.loads(args.kwargs) if args.kwargs else {}
    if call_args or call_kwargs or callable(obj) and not _is_circuit_like(obj):
        try:
            return obj(*call_args, **call_kwargs)
        except Exception as exc:
            raise SystemExit(
                f"Calling {args.target} with args={call_args} kwargs={call_kwargs} "
                f"failed: {type(exc).__name__}: {exc}"
            ) from exc
    return obj


def _is_circuit_like(obj: Any) -> bool:
    import cirq

    return isinstance(obj, cirq.AbstractCircuit)


def to_qasm(obj: Any, args: argparse.Namespace) -> str:
    """Lower a Bloq or Cirq circuit to OpenQASM 2."""
    import cirq

    if isinstance(obj, cirq.AbstractCircuit):
        print(f"Source: Cirq circuit, {len(obj.all_qubits())} qubits")
        print(f"Op census (as given): {op_census(obj)}")
        interceptor = None
        if args.unitary_uncompute:
            from ftcircuitbench.frontends import unitary_uncompute_interceptor

            interceptor = unitary_uncompute_interceptor
        return cirq_to_qasm2(
            obj,
            allow_non_unitary=args.allow_non_unitary,
            intercepting_decomposer=interceptor,
        )

    if not is_qualtran_available():
        raise SystemExit(
            f"Target resolved to {type(obj).__name__}, which is neither a Cirq "
            "circuit nor (without qualtran installed) a recognised Bloq."
        )
    from qualtran import Bloq

    if not isinstance(obj, Bloq):
        raise SystemExit(
            f"Target resolved to {type(obj).__name__}; expected a Qualtran Bloq "
            "or a Cirq circuit."
        )
    print(f"Source: Qualtran {type(obj).__name__}")
    circuit = bloq_to_cirq(obj, unitary_uncompute=args.unitary_uncompute)
    print(f"Decomposed to {len(circuit.all_qubits())} qubits")
    print(f"Op census: {op_census(circuit)}")
    return cirq_to_qasm2(
        circuit, decompose=False, allow_non_unitary=args.allow_non_unitary
    )


def default_output_path(target: str) -> str:
    attribute = target.partition(":")[2] or "circuit"
    return os.path.join("qasm", "imported", f"{attribute}.qasm")


def main() -> int:
    args = parse_arguments()

    if not is_cirq_available():
        print(CIRQ_INSTALL_HINT, file=sys.stderr)
        return 1

    obj = build_circuit_object(args)
    try:
        qasm = to_qasm(obj, args)
    except NonUnitaryCircuitError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    output = args.output or default_output_path(args.target)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(qasm)

    from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS, loads

    circuit = loads(qasm, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    counts = circuit.count_ops()
    print_table(
        "Exported OpenQASM 2",
        ["Metric", "Value"],
        [
            ["Path", output],
            ["Qubits", circuit.num_qubits],
            ["Instructions", len(circuit.data)],
            ["Depth", circuit.depth()],
            ["T family", counts.get("t", 0) + counts.get("tdg", 0)],
        ],
        ["<", ">"],
    )

    if args.analyze:
        from analyze_circuit import run_analysis

        print()
        run_analysis(
            qasm_file=output,
            pipeline=args.pipeline,
            gridsynth_precision=args.gridsynth_precision,
            skip_fidelity=True,
        )
    else:
        print(f"\nNext: uv run python analyze_circuit.py {output} --pipeline gs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
