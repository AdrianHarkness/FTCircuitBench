#!/usr/bin/env python3
"""CLI: physical resource estimates for FTCircuitBench logical counts.

Feeds Clifford+T (or PBC) counts to the Azure Quantum Resource Estimator and
reports physical qubits, runtime, and code distance per hardware model. See
`ftcircuitbench.resource_estimation.azure_qre` for the library entry points.
"""

import argparse
import fnmatch
import os
import sys
from typing import List, Optional, Sequence

from ftcircuitbench.benchmark_utils import format_time, print_table, save_json
from ftcircuitbench.resource_estimation import (
    DEFAULT_MODEL_NAMES,
    HARDWARE_MODELS,
    CircuitResourceEstimate,
    LogicalResourceCounts,
    estimate_circuits,
    is_qre_available,
    read_ct_stats_csv,
    read_stats_json,
    resolve_hardware_models,
)
from ftcircuitbench.resource_estimation.azure_qre import QRE_INSTALL_HINT

DEFAULT_CT_STATS = os.path.join("circuit_benchmarks", "ct_stats.csv")
DEFAULT_OUTPUT = os.path.join("qre_output", "qre_results.json")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the resource estimator."""
    parser = argparse.ArgumentParser(
        description="FTCircuitBench -> Azure Quantum Resource Estimator bridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--ct-stats",
        default=DEFAULT_CT_STATS,
        help="Aggregated Clifford+T stats CSV produced by generate_benchmarks.py.",
    )
    source.add_argument(
        "--stats-json",
        nargs="+",
        metavar="PATH",
        help="One or more per-circuit *_stats.json files written by "
        "analyze_circuit.py, used instead of the aggregated CSV.",
    )
    parser.add_argument(
        "--counts",
        choices=["clifford_t", "pbc"],
        default="clifford_t",
        help="Which logical counts to feed the estimator. 'pbc' uses the "
        "post-optimization PBC rotation/measurement operator counts and is only "
        "available with --stats-json.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="NAME",
        help="Hardware model to estimate under; repeatable. "
        f"Default: {', '.join(DEFAULT_MODEL_NAMES)}. "
        "Use --list-models to see the available names.",
    )
    parser.add_argument(
        "--error-budget",
        type=float,
        default=None,
        help="Override the total error budget of every selected model.",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        metavar="PATTERN",
        help="Only estimate circuits whose label matches this glob pattern "
        "(e.g. 'heisenberg-1d-100q-*'); repeatable. Ignored with --stats-json.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path for the JSON results file.",
    )
    parser.add_argument(
        "--summary-model",
        default=None,
        help="Model to tabulate in the terminal summary. Defaults to the first "
        "selected model.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the available hardware models and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-circuit progress trace.",
    )
    return parser.parse_args()


def print_models() -> None:
    rows = [
        [
            model.name,
            model.qubit_params,
            model.qec_scheme or "(QRE default)",
            f"{model.error_budget:g}",
        ]
        for model in HARDWARE_MODELS.values()
    ]
    print_table(
        "Available Hardware Models",
        ["Name", "Qubit Params", "QEC Scheme", "Error Budget"],
        rows,
    )


def _filter_labels(
    counts: Sequence[LogicalResourceCounts], patterns: Optional[Sequence[str]]
) -> List[LogicalResourceCounts]:
    if not patterns:
        return list(counts)
    return [
        item
        for item in counts
        if any(fnmatch.fnmatch(item.label, pattern) for pattern in patterns)
    ]


def _drop_zero_t(
    counts: Sequence[LogicalResourceCounts],
) -> List[LogicalResourceCounts]:
    """Drop Clifford-only circuits, which QRE has no T factory to estimate."""
    estimable = [item for item in counts if item.t_count > 0]
    skipped = [item.label for item in counts if item.t_count <= 0]
    if skipped:
        shown = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
        print(f"[WARN] Skipping {len(skipped)} circuit(s) with no T gates: {shown}")
    return estimable


def load_counts(args: argparse.Namespace) -> List[LogicalResourceCounts]:
    """Build the logical counts to estimate from the selected input source."""
    if args.stats_json:
        try:
            counts = [
                read_stats_json(path, counts_source=args.counts)
                for path in args.stats_json
            ]
        except (KeyError, ValueError, OSError) as exc:
            raise SystemExit(f"Could not read stats JSON: {exc}") from exc
        return _drop_zero_t(counts)

    if args.counts == "pbc":
        raise SystemExit(
            "--counts pbc needs per-circuit PBC statistics; pass --stats-json "
            "with one or more *_stats.json files."
        )
    if not os.path.exists(args.ct_stats):
        raise SystemExit(
            f"Clifford+T stats CSV '{args.ct_stats}' not found. Run "
            "generate_benchmarks.py first, or pass --ct-stats/--stats-json."
        )
    try:
        rows = read_ct_stats_csv(args.ct_stats, skip_zero_t=False)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Could not read {args.ct_stats}: {exc}") from exc
    return _drop_zero_t(_filter_labels(rows, args.labels))


def print_summary(results: Sequence[CircuitResourceEstimate], model_name: str) -> None:
    """Tabulate one model's physical cost across every estimated circuit."""
    rows = []
    for result in results:
        estimate = result.estimates.get(model_name)
        if estimate is None:
            rows.append(
                [
                    result.label,
                    f"{result.counts.t_count:,}",
                    result.errors.get(model_name, "not estimated"),
                    "",
                    "",
                ]
            )
            continue
        rows.append(
            [
                result.label,
                f"{result.counts.t_count:,}",
                f"{estimate.physical_qubits:,}",
                format_time(estimate.runtime_seconds),
                estimate.code_distance,
            ]
        )
    print_table(
        f"Physical Resources ({model_name})",
        ["Circuit", "T count", "Physical qubits", "Runtime", "Code distance"],
        rows,
        ["<", ">", ">", ">", ">"],
    )


def main() -> int:
    args = parse_arguments()

    if args.list_models:
        print_models()
        return 0

    if not is_qre_available():
        print(QRE_INSTALL_HINT, file=sys.stderr)
        return 1

    try:
        models = resolve_hardware_models(args.models, error_budget=args.error_budget)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1

    counts = load_counts(args)
    if not counts:
        print("No circuits selected; nothing to estimate.", file=sys.stderr)
        return 1

    print("=== FTCircuitBench -> Azure QRE ===")
    print(f"Circuits: {len(counts)}")
    print(f"Counts source: {args.counts}")
    print(f"Models: {', '.join(model.name for model in models)}")

    results = estimate_circuits(counts, models=models, progress=not args.quiet)

    save_json([result.to_dict() for result in results], args.output)
    print(f"\nSaved estimates for {len(results)} circuits to: {args.output}")

    summary_model = args.summary_model or models[0].name
    if summary_model not in {model.name for model in models}:
        print(
            f"\n[WARN] --summary-model '{summary_model}' was not estimated; "
            f"showing '{models[0].name}' instead.",
            file=sys.stderr,
        )
        summary_model = models[0].name
    print_summary(results, summary_model)

    failures = sum(1 for result in results if result.errors)
    if failures:
        print(
            f"\n[WARN] {failures} circuit(s) had at least one model fail to estimate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
