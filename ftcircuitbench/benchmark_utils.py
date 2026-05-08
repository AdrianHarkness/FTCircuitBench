# benchmark_utils.py

import json
import numbers
import os
from typing import Any, Dict, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump

# --- Top-Level Utility Functions ---


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if not isinstance(seconds, (int, float)):
        return "N/A"
    if seconds < 0.001 and seconds != 0:
        return f"{seconds*1000:.3f}ms"
    if seconds < 0.01 and seconds != 0:
        return f"{seconds*1000:.2f}ms"
    if seconds < 1:
        return f"{seconds:.3f}s"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def percent_reduction(pre, post):  # <-- MOVED TO TOP LEVEL
    """Calculate the percentage reduction between two numbers."""
    pre = pre if isinstance(pre, numbers.Number) else 0
    post = post if isinstance(post, numbers.Number) else 0
    if pre == 0 and post == 0:
        return "0.00%"
    if pre == 0:
        return "N/A"
    return f"{((pre - post) / pre) * 100:.2f}%"


def fmt_float_cell(value: Any, digits: int = 2):
    """Format numeric values to a fixed number of decimals; pass through non-numerics.

    Returns a string for numbers; otherwise returns the original value (or "N/A" for None).
    """
    if isinstance(value, numbers.Number):
        try:
            return f"{float(value):.{digits}f}"  # type: ignore[arg-type]
        except Exception:
            return str(value)
    return "N/A" if value is None else value


def save_qasm_circuit(circuit: QuantumCircuit, filepath: str):
    """Save a Qiskit circuit to a QASM 2.0 file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            dump(circuit, f)
    except Exception as e:
        print(f"Error saving QASM to {filepath}: {e}")


def find_all_qasm_files(qasm_root="qasm") -> Dict[str, Dict[str, str]]:
    """Recursively finds all .qasm files in a root directory."""
    qasm_files: Dict[str, Dict[str, str]] = {}
    for dirpath, _, filenames in os.walk(qasm_root):
        for filename in filenames:
            if filename.endswith(".qasm"):
                rel_dir = os.path.relpath(dirpath, qasm_root)
                category = rel_dir if rel_dir != "." else "root"
                if category not in qasm_files:
                    qasm_files[category] = {}
                instance_name = os.path.splitext(filename)[0]
                qasm_files[category][instance_name] = os.path.join(dirpath, filename)
    return qasm_files


def combine_pbc_files_same_dir(target_dir: str, run_prefix: str, stage: str):
    """
    Combines _tlayers.txt and _measure_basis.txt into a single file named <run_prefix>_<stage>.txt, then deletes the originals.
    Example: qft_4q_gs_prec10_pbc_pre_opt.txt
    """
    combined_filepath = os.path.join(target_dir, f"{run_prefix}_{stage}.txt")
    tlayers_filepath = os.path.join(target_dir, f"{run_prefix}_{stage}_tlayers.txt")
    measure_basis_filepath = os.path.join(
        target_dir, f"{run_prefix}_{stage}_measure_basis.txt"
    )
    try:
        with open(combined_filepath, "w") as outfile:
            outfile.write(f"--- T-Layers ({stage.replace('_',' ')}) ---\n")
            if os.path.exists(tlayers_filepath):
                with open(tlayers_filepath, "r") as infile:
                    outfile.write(infile.read())
            else:
                outfile.write("File not found.\n")
            outfile.write("\n\n")
            outfile.write(f"--- Measurement Basis ({stage.replace('_',' ')}) ---\n")
            if os.path.exists(measure_basis_filepath):
                with open(measure_basis_filepath, "r") as infile:
                    outfile.write(infile.read())
            else:
                outfile.write("File not found.\n")
        # Only delete after successful write
        if os.path.exists(tlayers_filepath):
            os.remove(tlayers_filepath)
        if os.path.exists(measure_basis_filepath):
            os.remove(measure_basis_filepath)
    except Exception as e:
        print(f"[ERROR] Failed to combine PBC files: {e}")


def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [stringify_keys(i) for i in obj]
    else:
        return obj


def make_json_serializable(obj):
    """Recursively convert numpy types to Python-native types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_json(data, filepath):
    """
    Saves a dictionary or list to a JSON file, making it JSON serializable.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(make_json_serializable(data), f, indent=2)


# --- Reporting and Printing Utilities ---


def print_table(
    title: str,
    headers: list,
    rows: list,
    column_alignments: Optional[list] = None,
):
    """Print a formatted table."""
    if not rows:
        print(f"\n=== {title} ===")
        print("  (No data to display)")
        return
    num_cols = len(headers)
    if column_alignments is None:
        column_alignments = ["<"] * num_cols
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    format_parts = [
        f"{{:{align}{width}}}" for align, width in zip(column_alignments, col_widths)
    ]
    format_str = "  ".join(format_parts)
    print(f"\n=== {title} ===")
    print(format_str.format(*headers))
    separator = "  ".join(["-" * width for width in col_widths])
    print(separator)
    for row in rows:
        safe_row = ["N/A" if cell is None else cell for cell in row]
        formatted_row = [str(item) for item in safe_row]
        print(format_str.format(*formatted_row))


def print_circuit_stats(title: str, stats: Dict[str, Any], show_detailed: bool = False):
    """Prints a comprehensive, structured summary of all circuit statistics."""
    if "fidelity_method" not in stats and "method" in stats:
        stats["fidelity_method"] = stats["method"]
    print(f"\n{'='*10} {title} {'='*10}")
    if not stats:
        print("  (No statistics available)")
        return

    general_info_rows = get_general_info(stats)
    print_table(
        "General Circuit Info", ["Metric", "Value"], general_info_rows, ["<", ">"]
    )

    if "total_t_family_count" in stats:
        ct_rows = get_clifford_rows(stats)
        print_table("Clifford+T Gate Counts", ["Metric", "Value"], ct_rows, ["<", ">"])

    # Add Clifford+T interaction graph statistics
    if any(key.startswith("interaction_graph_") for key in stats.keys()):
        ct_interaction_rows = get_interaction_graph_rows(stats, "")
        if ct_interaction_rows:
            print_table(
                "Clifford+T Interaction Graph Statistics",
                ["Metric", "Value"],
                ct_interaction_rows,
                ["<", ">"],
            )

    if "pre_opt_rotation_operators" in stats:
        pbc_rows = []
        # Rotation operators row always shown
        pbc_rows.append(
            [
                "Rotation Operators",
                stats.get("pre_opt_rotation_operators"),
                stats.get("pbc_rotation_operators"),
                percent_reduction(
                    stats.get("pre_opt_rotation_operators"),
                    stats.get("pbc_rotation_operators"),
                ),
            ]
        )
        # Rotation layers only if both pre/post are present and numeric
        pre_layers = stats.get("pre_opt_rotation_layers")
        post_layers = stats.get("pbc_rotation_layers")
        if isinstance(pre_layers, (int, float)) and isinstance(
            post_layers, (int, float)
        ):
            pbc_rows.append(
                [
                    "Rotation Layers",
                    pre_layers,
                    post_layers,
                    percent_reduction(pre_layers, post_layers),
                ]
            )
        # Measurement operators
        pbc_rows.append(
            [
                "Measurement Operators",
                stats.get("pre_opt_measurement_operators"),
                stats.get("pbc_measurement_operators"),
                percent_reduction(
                    stats.get("pre_opt_measurement_operators"),
                    stats.get("pbc_measurement_operators"),
                ),
            ]
        )
        pbc_rows.append(["", "", "", ""])  # spacer
        # Pauli weight stats
        pbc_rows.extend(
            [
                [
                    "Avg Operator Pauli Weight",
                    fmt_float_cell(stats.get("pre_opt_avg_operator_pauli_weight", 0.0)),
                    fmt_float_cell(stats.get("pbc_avg_operator_pauli_weight", 0.0)),
                    percent_reduction(
                        stats.get("pre_opt_avg_operator_pauli_weight"),
                        stats.get("pbc_avg_operator_pauli_weight"),
                    ),
                ],
                [
                    "Std Operator Pauli Weight",
                    fmt_float_cell(stats.get("pre_opt_std_operator_pauli_weight", 0.0)),
                    fmt_float_cell(stats.get("pbc_std_operator_pauli_weight", 0.0)),
                    percent_reduction(
                        stats.get("pre_opt_std_operator_pauli_weight"),
                        stats.get("pbc_std_operator_pauli_weight"),
                    ),
                ],
                [
                    "Max Operator Pauli Weight",
                    stats.get("pre_opt_max_operator_pauli_weight", "N/A"),
                    stats.get("pbc_max_operator_pauli_weight", "N/A"),
                    percent_reduction(
                        stats.get("pre_opt_max_operator_pauli_weight"),
                        stats.get("pbc_max_operator_pauli_weight"),
                    ),
                ],
            ]
        )
        pbc_rows.append(["", "", "", ""])  # spacer
        # Qubit interaction degree stats
        pbc_rows.extend(
            [
                [
                    "Avg Qubit Interaction Degree",
                    fmt_float_cell(
                        stats.get("pre_opt_avg_qubit_interaction_degree", 0.0)
                    ),
                    fmt_float_cell(stats.get("pbc_avg_qubit_interaction_degree", 0.0)),
                    percent_reduction(
                        stats.get("pre_opt_avg_qubit_interaction_degree"),
                        stats.get("pbc_avg_qubit_interaction_degree"),
                    ),
                ],
                [
                    "Std Qubit Interaction Degree",
                    fmt_float_cell(
                        stats.get("pre_opt_std_qubit_interaction_degree", 0.0)
                    ),
                    fmt_float_cell(stats.get("pbc_std_qubit_interaction_degree", 0.0)),
                    percent_reduction(
                        stats.get("pre_opt_std_qubit_interaction_degree"),
                        stats.get("pbc_std_qubit_interaction_degree"),
                    ),
                ],
                [
                    "Max Qubit Interaction Degree",
                    stats.get("pre_opt_max_qubit_interaction_degree", "N/A"),
                    stats.get("pbc_max_qubit_interaction_degree", "N/A"),
                    percent_reduction(
                        stats.get("pre_opt_max_qubit_interaction_degree"),
                        stats.get("pbc_max_qubit_interaction_degree"),
                    ),
                ],
            ]
        )
        print_table(
            "PBC Circuit Optimization Summary",
            ["Metric", "Value Pre Opt", "Value Post Opt", "% Reduction"],
            pbc_rows,
            ["<", ">", ">", ">"],
        )

    # Add PBC interaction graph statistics
    if any(key.startswith("pbc_interaction_graph_") for key in stats.keys()):
        pbc_interaction_rows = get_interaction_graph_rows(stats, "pbc_")
        if pbc_interaction_rows:
            print_table(
                "PBC Interaction Graph Statistics",
                ["Metric", "Value"],
                pbc_interaction_rows,
                ["<", ">"],
            )

    timing_rows = []
    if "transpilation_clifford_t_time" in stats:
        timing_rows.append(
            [
                "Clifford+T Transpilation",
                format_time(stats["transpilation_clifford_t_time"]),
            ]
        )
    if "pbc_conversion_time" in stats:
        timing_rows.append(
            ["PBC Conversion (Total)", format_time(stats["pbc_conversion_time"])]
        )
    if "total_time" in stats:
        timing_rows.append(["Total Pipeline Time", format_time(stats["total_time"])])
    if timing_rows:
        print_table("Pipeline Timings", ["Stage", "Time"], timing_rows, ["<", ">"])


def print_pipeline_comparison(
    gs_full_stats: Dict[str, Any], sk_full_stats: Dict[str, Any]
):
    """Prints a formatted, side-by-side comparison of the Gridsynth and Solovay-Kitaev pipelines."""
    print("\n\n=============================================")
    print("===          PIPELINE COMPARISON          ===")
    print("=============================================")

    headers = ["Metric", "Gridsynth Pipeline", "Solovay-Kitaev Pipeline"]
    align = ["<", ">", ">"]

    timing_rows = [
        [
            "Total Pipeline Time",
            format_time(gs_full_stats.get("total_time", 0)),
            format_time(sk_full_stats.get("total_time", 0)),
        ],
        [
            "  C+T Transpilation Time",
            format_time(gs_full_stats.get("transpilation_clifford_t_time", 0)),
            format_time(sk_full_stats.get("transpilation_clifford_t_time", 0)),
        ],
        [
            "  PBC Conversion Time",
            format_time(gs_full_stats.get("pbc_conversion_time", 0)),
            format_time(sk_full_stats.get("pbc_conversion_time", 0)),
        ],
    ]
    print_table("Overall Timing", headers, timing_rows, align)

    ct_rows = [
        [
            "C+T Total Gates",
            f"{gs_full_stats.get('total_gate_count', 'N/A'):,}",
            f"{sk_full_stats.get('total_gate_count', 'N/A'):,}",
        ],
        [
            "C+T Depth",
            f"{gs_full_stats.get('depth', 'N/A'):,}",
            f"{sk_full_stats.get('depth', 'N/A'):,}",
        ],
        [
            "C+T Total T-family",
            f"{gs_full_stats.get('total_t_family_count', 'N/A'):,}",
            f"{sk_full_stats.get('total_t_family_count', 'N/A'):,}",
        ],
        [
            "C+T Fidelity",
            (
                f"{gs_full_stats.get('fidelity', 'N/A'):.6e}"
                if isinstance(gs_full_stats.get("fidelity"), float)
                else "N/A"
            ),
            (
                f"{sk_full_stats.get('fidelity', 'N/A'):.6e}"
                if isinstance(sk_full_stats.get("fidelity"), float)
                else "N/A"
            ),
        ],
    ]
    print_table("Clifford+T Stage Comparison", headers, ct_rows, align)

    gs_init_t = gs_full_stats.get("initial_clifford_t_t_gates_for_pbc", 0)
    sk_init_t = sk_full_stats.get("initial_clifford_t_t_gates_for_pbc", 0)
    gs_opt_t = gs_full_stats.get("pbc_rotation_operators", 0)
    sk_opt_t = sk_full_stats.get("pbc_rotation_operators", 0)
    gs_reduction = ((gs_init_t - gs_opt_t) / gs_init_t * 100) if gs_init_t > 0 else 0
    sk_reduction = ((sk_init_t - sk_opt_t) / sk_init_t * 100) if sk_init_t > 0 else 0

    def _fmt_int(stats: dict, key: str) -> str:
        v = stats.get(key)
        return f"{v:,}" if isinstance(v, (int, float)) else "N/A"

    pbc_rows = [
        ["Input T-gates for PBC", f"{gs_init_t:,}", f"{sk_init_t:,}"],
        ["Final PBC Rotation Ops", f"{gs_opt_t:,}", f"{sk_opt_t:,}"],
        ["T-gate Reduction by PBC", f"{gs_reduction:.2f}%", f"{sk_reduction:.2f}%"],
        [
            "Final PBC Rotation Layers",
            _fmt_int(gs_full_stats, "pbc_rotation_layers"),
            _fmt_int(sk_full_stats, "pbc_rotation_layers"),
        ],
        [
            "Avg. Pauli Weight (Final)",
            f"{gs_full_stats.get('pbc_avg_operator_pauli_weight', 0.0):.2f}",
            f"{sk_full_stats.get('pbc_avg_operator_pauli_weight', 0.0):.2f}",
        ],
    ]
    print_table("PBC Optimization Comparison", headers, pbc_rows, align)


def get_general_info(stats):
    general_info = [
        ["Qubits", stats.get("num_qubits", "N/A")],
        ["Total Gates", stats.get("total_gate_count", "N/A")],
        ["Depth", stats.get("depth", "N/A")],
        [
            "Fidelity (vs Original)",
            (
                f"{stats.get('fidelity', 'N/A'):.6e}"
                if isinstance(stats.get("fidelity"), float)
                else "N/A"
            ),
        ],
        [
            "Fidelity Method",
            (
                stats.get("fidelity_method", "N/A").replace("_", " ").title()
                if isinstance(stats.get("fidelity_method"), str)
                else "N/A"
            ),
        ],
    ]
    if "gridsynth_precision" in stats and stats["gridsynth_precision"] is not None:
        general_info.append(
            ["Gridsynth Precision Used", stats.get("gridsynth_precision")]
        )
    if (
        "solovay_kitaev_recursion" in stats
        and stats["solovay_kitaev_recursion"] is not None
    ):
        general_info.append(
            ["Solovay-Kitaev Recursion", stats.get("solovay_kitaev_recursion")]
        )
    return general_info


def get_clifford_rows(stats):
    """Extract Clifford+T related statistics for display."""
    rows = []
    if "total_t_family_count" in stats:
        rows.append(["Total T-family gates", stats["total_t_family_count"]])
    if "t_count" in stats:
        rows.append(["T gates", stats["t_count"]])
    if "tdg_count" in stats:
        rows.append(["Tdg gates", stats["tdg_count"]])
    if "clifford_gate_count" in stats:
        rows.append(["Clifford gates", stats["clifford_gate_count"]])
    if "total_two_qubit_gates" in stats:
        rows.append(["Two-qubit gates", stats["total_two_qubit_gates"]])
    if "depth" in stats:
        rows.append(["Circuit depth", stats["depth"]])
    return rows


def get_interaction_graph_rows(stats, prefix=""):
    """Extract interaction graph statistics for display."""
    rows = []

    # Basic graph properties
    if f"{prefix}interaction_graph_num_nodes" in stats:
        rows.append(["Number of nodes", stats[f"{prefix}interaction_graph_num_nodes"]])
    if f"{prefix}interaction_graph_num_edges" in stats:
        rows.append(["Number of edges", stats[f"{prefix}interaction_graph_num_edges"]])
    # Skip graph density for PBC circuits as it's not a relevant statistic
    if not prefix.startswith("pbc_"):
        if f"{prefix}interaction_graph_density" in stats:
            density = stats[f"{prefix}interaction_graph_density"]
            if isinstance(density, (int, float)):
                rows.append(["Graph density", f"{density:.4f}"])
            else:
                rows.append(["Graph density", density])
    if f"{prefix}interaction_graph_is_connected" in stats:
        rows.append(["Is connected", stats[f"{prefix}interaction_graph_is_connected"]])
    if f"{prefix}interaction_graph_num_connected_components" in stats:
        rows.append(
            [
                "Connected components",
                stats[f"{prefix}interaction_graph_num_connected_components"],
            ]
        )

    # Degree statistics
    if f"{prefix}interaction_graph_avg_degree" in stats:
        avg_degree = stats[f"{prefix}interaction_graph_avg_degree"]
        if isinstance(avg_degree, (int, float)):
            rows.append(["Average degree", f"{avg_degree:.2f}"])
        else:
            rows.append(["Average degree", avg_degree])
    if f"{prefix}interaction_graph_std_degree" in stats:
        std_degree = stats[f"{prefix}interaction_graph_std_degree"]
        if isinstance(std_degree, (int, float)):
            rows.append(["Std dev degree", f"{std_degree:.2f}"])
        else:
            rows.append(["Std dev degree", std_degree])
    if f"{prefix}interaction_graph_min_degree" in stats:
        rows.append(["Min degree", stats[f"{prefix}interaction_graph_min_degree"]])
    if f"{prefix}interaction_graph_max_degree" in stats:
        rows.append(["Max degree", stats[f"{prefix}interaction_graph_max_degree"]])

    # Graph centrality measures
    if f"{prefix}interaction_graph_clustering_coefficient" in stats:
        cc = stats[f"{prefix}interaction_graph_clustering_coefficient"]
        if isinstance(cc, (int, float)):
            rows.append(["Clustering coefficient", f"{cc:.4f}"])
        else:
            rows.append(["Clustering coefficient", cc])
    if f"{prefix}interaction_graph_avg_shortest_path_length" in stats:
        aspl = stats[f"{prefix}interaction_graph_avg_shortest_path_length"]
        if isinstance(aspl, (int, float)):
            rows.append(["Avg shortest path length", f"{aspl:.2f}"])
        else:
            rows.append(["Avg shortest path length", aspl])
    if f"{prefix}interaction_graph_diameter" in stats:
        rows.append(["Diameter", stats[f"{prefix}interaction_graph_diameter"]])

    # Community metrics (strip 'louvain_' legacy prefix via fallbacks)
    mod_key = f"{prefix}interaction_graph_modularity"
    if mod_key in stats:
        modularity = stats[mod_key]
        if isinstance(modularity, (int, float)):
            rows.append(["Modularity", f"{modularity:.4f}"])
        else:
            rows.append(["Modularity", modularity])
    numc_key = f"{prefix}interaction_graph_num_communities"
    if numc_key in stats:
        rows.append(["Number of communities", stats[numc_key]])
    avgcs_key = f"{prefix}interaction_graph_avg_community_size"
    if avgcs_key in stats:
        avg_size = stats[avgcs_key]
        if isinstance(avg_size, (int, float)):
            rows.append(["Avg community size", f"{avg_size:.2f}"])
        else:
            rows.append(["Avg community size", avg_size])
    stdcs_key = f"{prefix}interaction_graph_std_community_size"
    if stdcs_key in stats:
        std_size = stats[stdcs_key]
        if isinstance(std_size, (int, float)):
            rows.append(["Std community size", f"{std_size:.2f}"])
        else:
            rows.append(["Std community size", std_size])
    mincs_key = f"{prefix}interaction_graph_min_community_size"
    if mincs_key in stats:
        rows.append(["Min community size", stats[mincs_key]])
    maxcs_key = f"{prefix}interaction_graph_max_community_size"
    if maxcs_key in stats:
        rows.append(["Max community size", stats[maxcs_key]])

    return rows


def get_output_param_str(pipeline: str, param_value: int, param_type: str) -> str:
    """
    Returns a standardized parameter string for output files.
    Example: 'gs_prec10' or 'sk_rec2'
    """
    if pipeline.lower() == "gs" or param_type == "precision_level":
        return f"gs_prec{param_value}"
    else:
        return f"sk_rec{param_value}"


def get_clifford_t_qasm_path(output_dir: str, circuit_name: str, param_str: str) -> str:
    """
    Returns the path for the Clifford+T QASM output file.
    """
    return os.path.join(output_dir, f"{circuit_name}_{param_str}_clifford_t.qasm")


def get_pbc_prefix(output_dir: str, circuit_name: str, param_str: str) -> str:
    """
    Returns the prefix for PBC output files (for T-layers, measure basis, etc.).
    """
    return os.path.join(output_dir, f"{circuit_name}_{param_str}_pbc")


def get_stats_json_path(output_dir: str, circuit_name: str, param_str: str) -> str:
    """
    Returns the path for the stats JSON file.
    """
    return os.path.join(output_dir, f"{circuit_name}_{param_str}_stats.json")
