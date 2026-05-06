# ./ftcircuitbench/transpilers/gs_transpiler.py
"""
Gridsynth-based transpilation to Clifford+T basis.
"""

from typing import Tuple, Union

from qiskit import QuantumCircuit, transpile
from tqdm import tqdm

from ftcircuitbench.decomposer import (  # Gridsynth decomposer
    decompose_rz_gates_gridsynth,
)
from ftcircuitbench.parser import (  # To load QASM if input is a path/string
    load_qasm_circuit,
)

# Define the intermediate basis for Gridsynth approach
GRIDSYNTH_INTERMEDIATE_BASIS = ["rz", "s", "h", "cx"]
# Note: 'x' and 'z' are Clifford. Gridsynth itself outputs S, H, T (and X if specified in the decomposer logic)
PBC_COMPATIBLE_CLIFFORD_T_BASIS = ["cx", "h", "s", "t", "tdg"]


def is_clifford_t_basis(circuit: QuantumCircuit) -> bool:
    """
    Check if a circuit is already in Clifford+T basis.

    Args:
        circuit (QuantumCircuit): The circuit to check

    Returns:
        bool: True if the circuit is in Clifford+T basis, False otherwise
    """
    # Get all unique gate names in the circuit
    gate_names = set()
    for instruction in circuit.data:
        gate_names.add(instruction.operation.name)

    # Check if all gates are in Clifford+T basis
    allowed_gates = set(PBC_COMPATIBLE_CLIFFORD_T_BASIS)
    return gate_names.issubset(allowed_gates)


def transpile_to_gridsynth_clifford_t(
    circuit_input: QuantumCircuit,
    is_file: bool = False,
    gridsynth_precision: int = 3,
    remove_final_measurements: bool = True,
    return_intermediate: bool = False,
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, QuantumCircuit]]:
    """
    Transpiles an input quantum circuit to a Clifford+T basis using Gridsynth for Rz decomposition.

    The process involves:
    1. (If input is QASM string/file) Loading the QASM.
    2. Removing final measurements (optional).
    3. Check if circuit is already in Clifford+T basis - if so, skip RZ transpilation.
    4. If not Clifford+T: Transpile the circuit to an intermediate basis: {'rz', 's', 'h', 'x', 'z', 'cx'}.
    5. Decompose all Rz gates into S, H, T sequences using Gridsynth.
    6. (Optional) A final validation to ensure the output is strictly in
       {'cx', 'h', 's', 't', 'tdg'} basis, making it suitable for PBC conversion.

    Args:
        circuit_input (QuantumCircuit | str): Input quantum circuit or QASM.
        is_file (bool): If circuit_input is a string, specifies if it's a file path.
        gridsynth_precision (int): Precision for Gridsynth Rz decomposition.
        remove_final_measurements (bool): If True, removes final measurements.
        return_intermediate (bool): If True, returns both intermediate and final circuits.

    Returns:
        QuantumCircuit or Tuple[QuantumCircuit, QuantumCircuit]: The final circuit, or (intermediate, final) if return_intermediate=True
    """
    if isinstance(circuit_input, str):
        initial_circuit = load_qasm_circuit(circuit_input, is_file=is_file)
    elif isinstance(circuit_input, QuantumCircuit):
        initial_circuit = circuit_input.copy()
    else:
        raise TypeError(
            "circuit_input must be a QuantumCircuit object or a QASM string/filepath."
        )

    if remove_final_measurements:
        initial_circuit.remove_final_measurements(inplace=True)

    # Check if circuit is already in Clifford+T basis
    if is_clifford_t_basis(initial_circuit):
        print(
            "      Circuit is already in Clifford+T basis. Skipping RZ transpilation."
        )
        if return_intermediate:
            return initial_circuit, initial_circuit
        else:
            return initial_circuit
    else:
        # Step 1: Transpile to intermediate RZ basis
        intermediate_circuit = transpile(
            initial_circuit,
            basis_gates=GRIDSYNTH_INTERMEDIATE_BASIS,
            optimization_level=0,  # Disable optimization for baseline
        )

        # Step 2: Decompose Rz gates using Gridsynth
        # Count Rz gates for progress bar
        rz_gates = [
            (i, inst, qargs)
            for i, (inst, qargs, _) in enumerate(intermediate_circuit.data)
            if inst.name == "rz"
        ]
        if not rz_gates:
            print("      No Rz gates found to decompose.")
            clifford_t_from_gs = intermediate_circuit
        else:
            with tqdm(
                total=len(rz_gates), desc="      Decomposing Rz gates", unit="gate"
            ) as pbar:
                clifford_t_from_gs = decompose_rz_gates_gridsynth(
                    intermediate_circuit,
                    precision=gridsynth_precision,
                    progress_bar=pbar,
                )

    # Step 3: Final validation to ensure PBC-compatible basis if requested

    allowed_gates = set(PBC_COMPATIBLE_CLIFFORD_T_BASIS)
    gate_names = {instruction.operation.name for instruction in clifford_t_from_gs.data}
    unexpected_gates = sorted(gate_names - allowed_gates)
    if unexpected_gates:
        clifford_t_from_gs = transpile(
            clifford_t_from_gs,
            basis_gates=PBC_COMPATIBLE_CLIFFORD_T_BASIS,
            optimization_level=0,
        )
    if return_intermediate:
        return intermediate_circuit, clifford_t_from_gs
    else:
        return clifford_t_from_gs
