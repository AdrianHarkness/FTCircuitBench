# analyze_circuit.py (Refactored to use ftcircuitbench.api)
import argparse
import os
from typing import Dict, List, Optional

from ftcircuitbench.api import PipelineConfig, run_analysis_for_file
from ftcircuitbench.benchmark_utils import (
    get_clifford_t_qasm_path,
    get_output_param_str,
    get_pbc_prefix,
    get_stats_json_path,
    print_circuit_stats,
    print_pipeline_comparison,
    save_json,
)


def parse_arguments():
    """Parses command-line arguments for the circuit analyzer."""
    parser = argparse.ArgumentParser(
        description="FTCircuitBench Circuit Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("qasm_file", help="Path to the QASM file to analyze")
    parser.add_argument(
        "--gridsynth-precision",
        type=int,
        default=5,
        help="Gridsynth precision level (decimal digits): synthesis error per Rz "
        "is ~10^(-precision). Higher = more accurate but more T gates per Rz.",
    )
    parser.add_argument(
        "--sk-recursion",
        type=int,
        default=2,
        help="Solovay-Kitaev recursion depth: each level roughly squares the "
        "approximation accuracy. Higher = more accurate but more T gates per Rz.",
    )
    parser.add_argument(
        "--layering-method",
        choices=["bare", "v2", "singleton"],
        default="v2",
        help="PBC layering strategy.",
    )
    parser.add_argument(
        "--layering-max-checks",
        type=int,
        default=None,
        help="Bound v2 layering to the last K layers when scanning for insertion (ignored for other methods)",
    )
    parser.add_argument(
        "--pipeline",
        choices=["gs", "sk", "both"],
        default="gs",
        help="Which Clifford+T transpilation pipeline to run.",
    )
    parser.add_argument(
        "--gs-backend",
        choices=["auto", "cpp", "python"],
        default="auto",
        help="Gridsynth backend: 'cpp' uses the nwqec C++ implementation, 'python' "
        "forces the Python implementation, 'auto' (default) prefers cpp when "
        "nwqec is installed. Ignored for the SK pipeline.",
    )
    parser.add_argument(
        "--pbc-backend",
        choices=["auto", "cpp", "python"],
        default="auto",
        help="PBC conversion backend: 'cpp' uses the nwqec C++ PBC adapter, "
        "'python' forces the Python `RotationPauliCirc` + layering pass, 'auto' "
        "(default) uses cpp when nwqec is installed. The Python path exposes "
        "layer-count metrics (`pbc_rotation_layers`) that the C++ path doesn't.",
    )
    parser.add_argument(
        "--optimize-t-maxiter",
        type=int,
        default=5,
        help="Number of T optimization iterations in the PBC stage (ignored unless --optimize-pbc)",
    )
    parser.add_argument(
        "--optimize-pbc",
        action="store_true",
        help="Enable PBC Tfuse/T-merging optimization",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum workers for parallel PBC conversion",
    )
    parser.add_argument(
        "--skip-fidelity",
        action="store_true",
        help="Skip fidelity calculation",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-stage tables in the pipeline output.",
    )
    return parser.parse_args()


def _build_pipeline_jobs(
    input_base: str,
    pipeline: str,
    gridsynth_precision: int,
    sk_recursion: int,
    layering_method: str,
    layering_max_checks: Optional[int],
    optimize_pbc: bool,
    optimize_t_maxiter: int,
    max_workers: Optional[int],
    calculate_fidelity: bool,
    gs_backend: str = "auto",
    pbc_backend: str = "auto",
) -> List[Dict[str, object]]:
    use_nwqec_pbc = pbc_backend != "python"
    jobs: List[Dict[str, object]] = []
    if pipeline in ["gs", "both"]:
        param_str = get_output_param_str("gs", gridsynth_precision, "precision_level")
        pbc_prefix = get_pbc_prefix("pbc_output", input_base, param_str)
        jobs.append(
            {
                "config": PipelineConfig(
                    pipeline="gs",
                    gridsynth_precision=gridsynth_precision,
                    layering_method=layering_method,
                    layering_max_checks=layering_max_checks,
                    optimize_pbc=optimize_pbc,
                    optimize_t_maxiter=optimize_t_maxiter,
                    return_intermediate=True,
                    calculate_fidelity=calculate_fidelity,
                    max_workers=max_workers,
                    prefer_cpp=(gs_backend != "python"),
                    use_nwqec_pbc=use_nwqec_pbc,
                    clifford_output_path=get_clifford_t_qasm_path(
                        "clifford_t_output", input_base, param_str
                    ),
                    pbc_output_prefix=pbc_prefix,
                ),
                "param_str": param_str,
                "stats_path": get_stats_json_path(
                    "circuit_stats_output", input_base, param_str
                ),
                "pbc_prefix": pbc_prefix,
            }
        )

    if pipeline in ["sk", "both"]:
        param_str = get_output_param_str("sk", sk_recursion, "recursion_degree")
        pbc_prefix = get_pbc_prefix("pbc_output", input_base, param_str)
        jobs.append(
            {
                "config": PipelineConfig(
                    pipeline="sk",
                    sk_recursion=sk_recursion,
                    layering_method=layering_method,
                    layering_max_checks=layering_max_checks,
                    optimize_pbc=optimize_pbc,
                    optimize_t_maxiter=optimize_t_maxiter,
                    return_intermediate=True,
                    calculate_fidelity=calculate_fidelity,
                    max_workers=max_workers,
                    use_nwqec_pbc=use_nwqec_pbc,
                    clifford_output_path=get_clifford_t_qasm_path(
                        "clifford_t_output", input_base, param_str
                    ),
                    pbc_output_prefix=pbc_prefix,
                ),
                "param_str": param_str,
                "stats_path": get_stats_json_path(
                    "circuit_stats_output", input_base, param_str
                ),
                "pbc_prefix": pbc_prefix,
            }
        )
    return jobs


def _summarize_pipeline_result(result, config: PipelineConfig) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    summary.update(result.clifford_stats)
    summary.update(result.pbc_stats)
    if result.fidelity:
        summary.update(result.fidelity)
        summary["fidelity_method"] = result.fidelity.get("method", "N/A")

    summary["pipeline"] = config.pipeline.upper()
    summary["gridsynth_precision"] = (
        config.gridsynth_precision if config.pipeline == "gs" else None
    )
    summary["solovay_kitaev_recursion"] = (
        config.sk_recursion if config.pipeline == "sk" else None
    )
    summary["pbc_layering_method"] = result.parameters.get("layering_method")
    summary["pbc_layering_max_checks"] = result.parameters.get("layering_max_checks")
    summary["pbc_optimize_t_maxiter"] = result.parameters.get("optimize_t_maxiter")
    summary["pbc_max_workers"] = result.parameters.get("max_workers")

    summary["transpilation_clifford_t_time"] = result.timings.get(
        "transpilation_clifford_t_time"
    )
    summary["pbc_conversion_time"] = result.timings.get("pbc_conversion_time")
    numeric_timings = [
        t for t in result.timings.values() if isinstance(t, (int, float))
    ]
    if numeric_timings:
        summary["total_time"] = sum(numeric_timings)
    return summary


def run_analysis(
    qasm_file: str,
    gridsynth_precision: int = 3,
    sk_recursion: int = 2,
    layering_method: str = "v2",
    layering_max_checks: Optional[int] = None,
    pipeline: str = "gs",
    optimize_pbc: bool = False,
    optimize_t_maxiter: int = 5,
    detailed: bool = False,
    max_workers: Optional[int] = None,
    skip_fidelity: bool = False,
    gs_backend: str = "auto",
    pbc_backend: str = "auto",
):
    """
    Programmatic API to run FTCircuitBench analysis using the packaged API.

    Args mirror the CLI flags; see docs/examples.md for usage.
    """
    if not os.path.exists(qasm_file):
        raise FileNotFoundError(f"QASM file '{qasm_file}' not found.")

    from ftcircuitbench.transpilers import is_nwqec_available

    if gs_backend == "cpp" and pipeline in ("gs", "both") and not is_nwqec_available():
        raise RuntimeError(
            "--gs-backend=cpp requested, but nwqec is not installed. "
            "Install nwqec or use --gs-backend=auto/python."
        )
    if pbc_backend == "cpp" and not is_nwqec_available():
        raise RuntimeError(
            "--pbc-backend=cpp requested, but nwqec is not installed. "
            "Install nwqec or use --pbc-backend=auto/python."
        )

    input_base = os.path.splitext(os.path.basename(qasm_file))[0]
    print("=== FTCircuitBench Analysis ===")
    print(f"Input: {qasm_file}")
    print(f"PBC Optimization: {'ON' if optimize_pbc else 'OFF'}")
    if pipeline in ("gs", "both"):
        print(f"GS backend: {gs_backend}")
    print(f"PBC backend: {pbc_backend}")

    jobs = _build_pipeline_jobs(
        input_base=input_base,
        pipeline=pipeline,
        gridsynth_precision=gridsynth_precision,
        sk_recursion=sk_recursion,
        layering_method=layering_method,
        layering_max_checks=layering_max_checks,
        optimize_pbc=optimize_pbc,
        optimize_t_maxiter=optimize_t_maxiter,
        max_workers=max_workers,
        calculate_fidelity=not skip_fidelity,
        gs_backend=gs_backend,
        pbc_backend=pbc_backend,
    )

    configs = [job["config"] for job in jobs]
    analysis = run_analysis_for_file(qasm_file, configs)

    results: Dict[str, Dict[str, object]] = {}
    for job in jobs:
        pipeline_name = job["config"].pipeline
        result = analysis.pipelines.get(pipeline_name)
        if result is None:
            continue

        summary = _summarize_pipeline_result(result, job["config"])
        stats_path = job["stats_path"]

        print(
            f"\n\n{'='*50}\n"
            f"=== Pipeline: {pipeline_name.upper()} -> PBC ===\n"
            f"{'='*50}"
        )
        print_circuit_stats(
            f"{pipeline_name.upper()} Pipeline Full Analysis", summary, detailed
        )

        save_json(summary, stats_path)
        print(f"  Saved full stats to: {stats_path}")

        if result.artifacts and "clifford_t_qasm" in result.artifacts:
            print(f"  Clifford+T QASM: {result.artifacts['clifford_t_qasm']}")
        if result.artifacts and "pbc_post_opt_layers" in result.artifacts:
            print(
                f"  PBC artifacts written under: "
                f"{os.path.dirname(result.artifacts['pbc_post_opt_layers'])}"
            )

        results[pipeline_name] = summary

    if pipeline == "both" and "gs" in results and "sk" in results:
        print_pipeline_comparison(results["gs"], results["sk"])

    print("\n--- FTCircuitBench Analysis Complete ---")
    return results


def main():
    """Entry point for CLI use."""
    args = parse_arguments()
    run_analysis(
        qasm_file=args.qasm_file,
        gridsynth_precision=args.gridsynth_precision,
        sk_recursion=args.sk_recursion,
        layering_method=args.layering_method,
        layering_max_checks=args.layering_max_checks,
        pipeline=args.pipeline,
        optimize_pbc=args.optimize_pbc,
        optimize_t_maxiter=args.optimize_t_maxiter,
        detailed=args.detailed,
        max_workers=args.max_workers,
        skip_fidelity=args.skip_fidelity,
        gs_backend=args.gs_backend,
        pbc_backend=args.pbc_backend,
    )


if __name__ == "__main__":
    main()
