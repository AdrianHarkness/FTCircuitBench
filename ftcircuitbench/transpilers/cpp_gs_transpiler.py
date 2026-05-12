# ./ftcircuitbench/transpilers/cpp_gs_transpiler.py
"""
C++-Gridsynth wrapper that delegates to nwqec's transpiler when available and
falls back to the Python Gridsynth path otherwise.

The actual pipeline shape is enforced inside the delegated transpilers
(`transpile_to_clifford_t_cpp` and `transpile_to_gridsynth_clifford_t`), which
both use the canonical staging from `_basis.py`.
"""

from typing import Tuple, Union

from qiskit import QuantumCircuit

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
    """Run Gridsynth-based Clifford+T transpilation via nwqec C++ when available."""
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
        raise RuntimeError(
            "nwqec C++ backend not available and fallback_to_python=False"
        )

    epsilon = 10.0 ** (-gridsynth_precision)
    try:
        print("      Using nwqec C++ backend for Clifford+T synthesis...")
        return nwqec_transpile_to_clifford_t(
            circuit_input=circuit_input,
            is_file=is_file,
            epsilon=epsilon,
            keep_ccx=False,
            remove_final_measurements=remove_final_measurements,
            forbid_python_fallback=True,
            return_intermediate=return_intermediate,
        )
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
        raise RuntimeError(f"nwqec C++ backend failed: {e}")


def is_cpp_gs_available() -> bool:
    """Check if nwqec C++ Gridsynth backend is available."""
    return is_nwqec_available()
