# ./ftcircuitbench/transpilers/cpp_gs_transpiler.py
"""
C++-based Gridsynth transpiler for FTCircuitBench.

This module provides a high-performance C++ implementation of Gridsynth-based
Clifford+T transpilation via nwqec, falling back to Python implementation if needed.
"""

from typing import Tuple, Union

from qiskit import QuantumCircuit

from ftcircuitbench.parser import load_qasm_circuit
from ftcircuitbench.transpilers.gs_transpiler import (
    is_clifford_t_basis,
)
from ftcircuitbench.transpilers.gs_transpiler import (
    transpile_to_gridsynth_clifford_t as python_gs_transpiler,
)
from ftcircuitbench.transpilers.nwqec_ct import (
    is_nwqec_available,
)
from ftcircuitbench.transpilers.nwqec_ct import (
    transpile_to_clifford_t_cpp as nwqec_transpile_to_clifford_t,
)


def transpile_to_gridsynth_clifford_t_cpp(
    circuit_input: Union[QuantumCircuit, str],
    is_file: bool = False,
    gridsynth_precision: int = 3,
    remove_final_measurements: bool = True,
    return_intermediate: bool = False,
    fallback_to_python: bool = True,
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, QuantumCircuit]]:
    """
    C++-based transpilation to Clifford+T basis using Gridsynth via nwqec.

    This function uses the high-performance nwqec C++ backend when available,
    falling back to the Python implementation if needed.

    Args:
        circuit_input: Input quantum circuit or QASM string/filepath
        is_file: If circuit_input is a string, specifies if it's a file path
        gridsynth_precision: Precision for Gridsynth Rz decomposition (converted to epsilon)
        remove_final_measurements: If True, removes final measurements
        return_intermediate: If True, returns both intermediate and final circuits
        fallback_to_python: If True, fall back to Python implementation if C++ fails

    Returns:
        QuantumCircuit or Tuple[QuantumCircuit, QuantumCircuit]: The final circuit, or (intermediate, final) if return_intermediate=True
    """
    # Check if nwqec C++ backend is available
    if not is_nwqec_available():
        if fallback_to_python:
            print(
                "      nwqec C++ backend not available, falling back to Python implementation..."
            )
            return python_gs_transpiler(
                circuit_input=circuit_input,
                is_file=is_file,
                gridsynth_precision=gridsynth_precision,
                remove_final_measurements=remove_final_measurements,
                return_intermediate=return_intermediate,
            )
        else:
            raise RuntimeError(
                "nwqec C++ backend not available and fallback_to_python=False"
            )

    # Load circuit if needed
    if isinstance(circuit_input, str):
        if is_file:
            initial_circuit = load_qasm_circuit(circuit_input, is_file=True)
        else:
            initial_circuit = load_qasm_circuit(circuit_input, is_file=False)
    elif isinstance(circuit_input, QuantumCircuit):
        initial_circuit = circuit_input.copy()
    else:
        raise TypeError(
            "circuit_input must be a QuantumCircuit object or a QASM string/filepath."
        )

    # Check if circuit is already in Clifford+T basis
    if is_clifford_t_basis(initial_circuit):
        print(
            "      Circuit is already in Clifford+T basis. Skipping RZ transpilation."
        )
        if return_intermediate:
            return initial_circuit, initial_circuit
        else:
            return initial_circuit

    try:
        # Use nwqec C++ transpiler
        print("      Using nwqec C++ backend for Clifford+T synthesis...")

        # Convert gridsynth_precision to epsilon (10^(-precision))
        epsilon = 10.0 ** (-gridsynth_precision)

        # Use nwqec transpiler
        result = nwqec_transpile_to_clifford_t(
            circuit_input=initial_circuit,
            is_file=False,
            epsilon=epsilon,
            keep_ccx=False,
            remove_final_measurements=remove_final_measurements,
            forbid_python_fallback=True,
            return_intermediate=return_intermediate,
        )

        if return_intermediate:
            intermediate, clifford_t_circuit = result
            return intermediate, clifford_t_circuit
        else:
            return result

    except Exception as e:
        if fallback_to_python:
            print(
                f"      nwqec C++ backend failed: {e}. Falling back to Python implementation..."
            )
            return python_gs_transpiler(
                circuit_input=circuit_input,
                is_file=is_file,
                gridsynth_precision=gridsynth_precision,
                remove_final_measurements=remove_final_measurements,
                return_intermediate=return_intermediate,
            )
        else:
            raise RuntimeError(f"nwqec C++ backend failed: {e}")


def is_cpp_gs_available() -> bool:
    """Check if nwqec C++ Gridsynth backend is available."""
    return is_nwqec_available()
