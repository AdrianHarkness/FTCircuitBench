# ./ftcircuitbench/decomposer/__init__.py
"""
Decomposition module for FTCircuitBench.
Focuses on decomposing RZ gates into Clifford+T or other desired gate sets.
"""

from .decomposer import (
    _run_gridsynth_cli,  # Exported for direct testing
    create_circuit_from_gate_string,
    decompose_rz_gates_gridsynth,
    decompose_rz_gates_pygridsynth,  # Placeholder
    parse_angle_from_gate_name,
)

__all__ = [
    "decompose_rz_gates_gridsynth",
    "create_circuit_from_gate_string",
    "parse_angle_from_gate_name",
    "decompose_rz_gates_pygridsynth",
    "_run_gridsynth_cli",
]
