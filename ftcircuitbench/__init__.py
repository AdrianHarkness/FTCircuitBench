# ./ftcircuitbench/__init__.py
"""
FTCircuitBench: A benchmark suite for fault-tolerant quantum circuit compilation and analysis.

The public API lives in :mod:`ftcircuitbench.api`, which exposes documented helpers for
running the Gridsynth (GS) and Solovay-Kitaev (SK) pipelines, converting to PBC, and
collecting fidelity and analysis metrics.
"""

import warnings

from .analyzer.clifford_t_analyzer import analyze_clifford_t_circuit
from .analyzer.pbc_analyzer import analyze_pbc_circuit
from .analyzer.visualization import (
    get_interaction_statistics,
    show_clifford_t_interaction_graph,
    show_operator_weight_histogram,
    show_pbc_interaction_graph,
    show_qubit_pbc_operations_plot,
)
from .api import (
    AnalysisResult,
    PipelineConfig,
    PipelineResult,
    run_analysis,
    run_analysis_for_file,
    run_pipeline,
)
from .decomposer import decompose_rz_gates_gridsynth
from .fidelity import (
    MAX_QUBITS_FOR_FIDELITY,
    calculate_circuit_fidelity,
    rz_product_fidelity,
)
from .parser import load_qasm_circuit, transpile_qasm_to_target_basis

# Import from pbc_converter, ensuring only publicly intended functions are listed
from .pbc_converter import convert_to_pbc_circuit  # get_pbc_stats is removed
from .transpilers import (
    transpile_to_gridsynth_clifford_t,
    transpile_to_solovay_kitaev_clifford_t,
)

# Suppress noisy NumPy det runtime warnings globally (applies to spawned processes too)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"numpy\.linalg\._linalg"
)
warnings.filterwarnings(
    "ignore",
    message=r".*(divide by zero|invalid value) encountered in det.*",
    category=RuntimeWarning,
)

__all__ = [
    # Public API surface
    "AnalysisResult",
    "PipelineConfig",
    "PipelineResult",
    "run_analysis",
    "run_analysis_for_file",
    "run_pipeline",
    # From parser
    "load_qasm_circuit",
    "transpile_qasm_to_target_basis",
    # From decomposer
    "decompose_rz_gates_gridsynth",
    # From transpilers
    "transpile_to_solovay_kitaev_clifford_t",
    "transpile_to_gridsynth_clifford_t",
    "analyze_clifford_t_circuit",
    "analyze_pbc_circuit",
    # From pbc_converter
    "convert_to_pbc_circuit",  # This now returns stats, so get_pbc_stats is not needed here
    # "get_pbc_stats", # Removed
    # From fidelity
    "rz_product_fidelity",
    "calculate_circuit_fidelity",
    "MAX_QUBITS_FOR_FIDELITY",
    # From analyzer
    "show_clifford_t_interaction_graph",
    "show_pbc_interaction_graph",
    "show_operator_weight_histogram",
    "show_qubit_pbc_operations_plot",
    "get_interaction_statistics",
]
