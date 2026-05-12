# ./ftcircuitbench/transpilers/__init__.py
"""
Transpilation utilities for FTCircuitBench.

This module provides transpilation to Clifford+T basis using:
- nwqec C++ backend (when available) for high-performance Gridsynth-based synthesis
- Python Gridsynth implementation (fallback)
- Solovay-Kitaev algorithm (Python only)
"""

from __future__ import annotations

from typing import Tuple, Union

from qiskit import QuantumCircuit

# Import nwqec PBC transpiler
from ..pbc_converter.nwqec_adapter import transpile_to_pbc_cpp

# Import Python-based transpilers (fallback)
from .gs_transpiler import transpile_to_gridsynth_clifford_t as _python_gs_transpiler
from .nwqec_ct import is_nwqec_available
from .nwqec_ct import is_nwqec_available as _is_nwqec_available
from .nwqec_ct import transpile_to_clifford_t_cpp
from .nwqec_ct import transpile_to_clifford_t_cpp as _nwqec_ct_transpiler
from .sk_transpiler import (
    transpile_to_solovay_kitaev_clifford_t as _python_sk_transpiler,
)

# Try to import C++ Gridsynth wrapper (uses nwqec internally)
try:
    from .cpp_gs_transpiler import (
        is_cpp_gs_available,
        transpile_to_gridsynth_clifford_t_cpp,
    )

    CPP_TRANSPILERS_AVAILABLE = True
except ImportError:
    CPP_TRANSPILERS_AVAILABLE = False

    # Create dummy functions for when C++ transpilers are not available.
    # Signatures intentionally match the real implementations so mypy treats
    # the conditional branches as compatible.
    def transpile_to_gridsynth_clifford_t_cpp(
        circuit_input: Union[QuantumCircuit, str],
        is_file: bool = False,
        gridsynth_precision: int = 3,
        remove_final_measurements: bool = True,
        return_intermediate: bool = False,
        fallback_to_python: bool = True,
    ) -> Union[QuantumCircuit, Tuple[QuantumCircuit, QuantumCircuit]]:
        raise ImportError(
            "C++ transpilers not available. Please install nwqec with C++ backend."
        )

    def is_cpp_gs_available() -> bool:
        return False


# Create the main transpiler functions that automatically use C++ when available
def transpile_to_gridsynth_clifford_t(
    circuit_input: Union[QuantumCircuit, str],
    is_file: bool = False,
    gridsynth_precision: int = 3,
    remove_final_measurements: bool = True,
    return_intermediate: bool = False,
    prefer_cpp: bool = True,
    force_python: bool = False,
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, QuantumCircuit]]:
    """
    Transpiles an input quantum circuit to a Clifford+T basis using Gridsynth for Rz decomposition.

    This function automatically uses the high-performance nwqec C++ backend when available,
    falling back to the Python implementation if needed.

    Args:
        circuit_input: Input quantum circuit or QASM string/filepath
        is_file: If circuit_input is a string, specifies if it's a file path
        gridsynth_precision: Precision for Gridsynth Rz decomposition
        remove_final_measurements: If True, removes final measurements
        return_intermediate: If True, returns both intermediate and final circuits
        prefer_cpp: If True (default), prefer nwqec C++ implementation when available
        force_python: If True, force Python path even if nwqec is available

    Returns:
        QuantumCircuit or Tuple[QuantumCircuit, QuantumCircuit]: The final circuit, or (intermediate, final) if return_intermediate=True
    """
    # Prefer nwqec C++ backend when available and not explicitly forced to Python
    if not force_python and prefer_cpp and _is_nwqec_available():
        epsilon = 10.0 ** (-gridsynth_precision)
        return _nwqec_ct_transpiler(
            circuit_input=circuit_input,
            is_file=is_file,
            epsilon=epsilon,
            keep_ccx=False,
            remove_final_measurements=remove_final_measurements,
            forbid_python_fallback=False,  # Allow fallback within nwqec
            return_intermediate=return_intermediate,
        )

    # Fallback to Python implementation
    return _python_gs_transpiler(
        circuit_input=circuit_input,
        is_file=is_file,
        gridsynth_precision=gridsynth_precision,
        remove_final_measurements=remove_final_measurements,
        return_intermediate=return_intermediate,
    )


def transpile_to_clifford_t_fast(
    circuit_input: Union[QuantumCircuit, str],
    is_file: bool = False,
    epsilon: float | None = None,
    keep_ccx: bool = False,
    remove_final_measurements: bool = True,
    prefer_cpp: bool = True,
    force_python: bool = False,
    return_intermediate: bool = False,
):
    """
    Fast Clifford+T transpilation using nwqec C++ backend when available.

    This is an alias for direct nwqec access with epsilon-based precision control.
    Falls back to Python Gridsynth if nwqec is not available.

    Args:
        circuit_input: Input quantum circuit or QASM string/filepath
        is_file: If circuit_input is a string, specifies if it's a file path
        epsilon: Target precision (error tolerance) for synthesis
        keep_ccx: If True, keep Toffoli gates instead of decomposing
        remove_final_measurements: If True, removes final measurements
        prefer_cpp: If True, prefer nwqec C++ implementation when available
        return_intermediate: If True, returns both intermediate and final circuits

    Returns:
        QuantumCircuit or Tuple[QuantumCircuit, QuantumCircuit]: The final circuit, or (intermediate, final) if return_intermediate=True
    """
    if not force_python and prefer_cpp and _is_nwqec_available():
        return _nwqec_ct_transpiler(
            circuit_input,
            is_file=is_file,
            epsilon=epsilon,
            keep_ccx=keep_ccx,
            remove_final_measurements=remove_final_measurements,
            forbid_python_fallback=False,
            return_intermediate=return_intermediate,
        )
    # Fallback to Python Gridsynth (convert epsilon to precision if needed)
    import math

    if epsilon is not None:
        gridsynth_precision = max(1, int(-math.log10(epsilon)))
    else:
        gridsynth_precision = 3
    return _python_gs_transpiler(
        circuit_input=circuit_input,
        is_file=is_file,
        gridsynth_precision=gridsynth_precision,
        remove_final_measurements=remove_final_measurements,
        return_intermediate=return_intermediate,
    )


def transpile_to_solovay_kitaev_clifford_t(
    circuit: Union[QuantumCircuit, str],
    recursion_degree: int = 3,
    remove_final_measurements: bool = True,
    return_intermediate: bool = False,
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, QuantumCircuit]]:
    """
    Transpiles an input QuantumCircuit to Solovay-Kitaev basis (Clifford+T).

    Note: Solovay-Kitaev is only available via Python implementation.
    nwqec does not provide Solovay-Kitaev synthesis.

    Args:
        circuit: Input quantum circuit or QASM string/filepath
        recursion_degree: The recursion degree for Solovay-Kitaev
        remove_final_measurements: If True, removes final measurements
        return_intermediate: If True, returns both intermediate and final circuits

    Returns:
        QuantumCircuit or Tuple[QuantumCircuit, QuantumCircuit]:
        The final circuit, or (intermediate_rz, final_sk) if return_intermediate=True
    """
    # Always use Python implementation for Solovay-Kitaev
    # (C++ transpiler only supports Gridsynth)
    return _python_sk_transpiler(
        circuit=circuit,
        recursion_degree=recursion_degree,
        remove_final_measurements=remove_final_measurements,
        return_intermediate=return_intermediate,
    )


__all__ = [
    # Main transpiler functions (auto-choose C++ or Python)
    "transpile_to_gridsynth_clifford_t",
    "transpile_to_solovay_kitaev_clifford_t",
    "transpile_to_clifford_t_fast",
    # Direct nwqec access
    "transpile_to_clifford_t_cpp",
    "transpile_to_pbc_cpp",
    # C++ transpiler wrapper (uses nwqec internally)
    "transpile_to_gridsynth_clifford_t_cpp",
    "is_cpp_gs_available",
    "CPP_TRANSPILERS_AVAILABLE",
    # Availability checks
    "is_nwqec_available",
]
