# ./ftcircuitbench/analyzer/__init__.py
"""
Analysis module for FTCircuitBench.
Provides functions to extract metrics from Clifford+T and PBC circuits.
"""

from .clifford_t_analyzer import analyze_clifford_t_circuit
from .pbc_analyzer import (  # Expose parser if useful externally
    analyze_pbc_circuit,
    parse_pbc_gate_name,
)
from .visualization import (
    get_interaction_statistics,
    show_clifford_t_interaction_graph,
    show_operator_weight_histogram,
    show_pbc_interaction_graph,
    show_qubit_pbc_operations_plot,
)

__all__ = [
    "analyze_clifford_t_circuit",
    "analyze_pbc_circuit",
    "parse_pbc_gate_name",
    "show_clifford_t_interaction_graph",
    "show_pbc_interaction_graph",
    "show_operator_weight_histogram",
    "show_qubit_pbc_operations_plot",
    "get_interaction_statistics",
]
