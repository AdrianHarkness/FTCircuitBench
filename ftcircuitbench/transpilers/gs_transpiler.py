# ./ftcircuitbench/transpilers/gs_transpiler.py
"""
Gridsynth-based transpilation to Clifford+T basis.

Pipeline shape is the canonical one defined in `_basis.py`. Only the RZ-synthesis
step is engine-specific (gridsynth here).
"""

from typing import Tuple, Union

from qiskit import QuantumCircuit
from tqdm import tqdm

from ftcircuitbench.decomposer import (  # Gridsynth decomposer
    decompose_rz_gates_gridsynth,
)

from ._basis import (
    INTERMEDIATE_RZ_BASIS,
    PBC_COMPATIBLE_CLIFFORD_T_BASIS,
    enforce_pbc_basis,
    is_clifford_t_basis,
    prepare_input,
    to_intermediate_rz,
)

GRIDSYNTH_INTERMEDIATE_BASIS = list(INTERMEDIATE_RZ_BASIS)


def transpile_to_gridsynth_clifford_t(
    circuit_input: Union[QuantumCircuit, str],
    is_file: bool = False,
    gridsynth_precision: int = 3,
    remove_final_measurements: bool = True,
    return_intermediate: bool = False,
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, QuantumCircuit]]:
    """Transpile to Clifford+T using Gridsynth for Rz decomposition.

    Pipeline:
      1. prepare_input (QC/QASM, optional measurement removal)
      2. early-return if already in PBC Clifford+T basis
      3. to_intermediate_rz (canonical {cx, h, s, rz} basis)
      4. gridsynth-based Rz decomposition
      5. enforce_pbc_basis ({cx, h, s, t, tdg})

    Returns the final circuit, or `(intermediate, final)` when
    `return_intermediate=True`. The intermediate is the canonical RZ-basis circuit
    (or the input itself when the early-return fires).
    """
    initial_circuit = prepare_input(
        circuit_input,
        is_file=is_file,
        remove_final_measurements=remove_final_measurements,
    )

    if is_clifford_t_basis(initial_circuit):
        print(
            "      Circuit is already in Clifford+T basis. Skipping RZ transpilation."
        )
        return (
            (initial_circuit, initial_circuit)
            if return_intermediate
            else initial_circuit
        )

    intermediate_circuit = to_intermediate_rz(initial_circuit)

    rz_gates = [
        (i, item.operation, item.qubits)
        for i, item in enumerate(intermediate_circuit.data)
        if item.operation.name == "rz"
    ]
    if not rz_gates:
        print("      No Rz gates found to decompose.")
        clifford_t_from_gs = intermediate_circuit
    else:
        with tqdm(
            total=len(rz_gates), desc="      Decomposing Rz gates", unit="gate"
        ) as pbar:
            clifford_t_from_gs, decomp_map = decompose_rz_gates_gridsynth(
                intermediate_circuit,
                precision=gridsynth_precision,
                progress_bar=pbar,
                return_decomp_map=True,
            )
        if intermediate_circuit.metadata is None:
            intermediate_circuit.metadata = {}
        intermediate_circuit.metadata["gridsynth_decomp"] = decomp_map
        intermediate_circuit.metadata["gridsynth_precision"] = gridsynth_precision

    clifford_t_from_gs = enforce_pbc_basis(clifford_t_from_gs)
    return (
        (intermediate_circuit, clifford_t_from_gs)
        if return_intermediate
        else clifford_t_from_gs
    )


__all__ = [
    "GRIDSYNTH_INTERMEDIATE_BASIS",
    "PBC_COMPATIBLE_CLIFFORD_T_BASIS",
    "is_clifford_t_basis",
    "transpile_to_gridsynth_clifford_t",
]
