# ./ftcircuitbench/transpilers/sk_transpiler.py
"""
Transpilation to Solovay-Kitaev basis (cx, h, s, t, tdg).
"""

import warnings
from typing import Tuple, Union

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dump  # For saving to QASM, if needed as utility
from qiskit.synthesis import generate_basic_approximations
from qiskit.transpiler.passes.synthesis import SolovayKitaev

# Define the target single-qubit basis for Solovay-Kitaev and subsequent PBC processing
SOLOVAY_KITAEV_BASIS = ["h", "s", "t", "tdg"]
# Intermediate basis before SK, ensuring Rz gates are present for SK to act upon
INTERMEDIATE_RZ_BASIS = ["cx", "h", "s", "rz"]


def transpile_to_solovay_kitaev_clifford_t(
    circuit: QuantumCircuit,
    recursion_degree: int = 3,
    remove_final_measurements: bool = True,
    return_intermediate: bool = False,
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, QuantumCircuit]]:
    """
    Transpiles an input QuantumCircuit.
    1. First to an intermediate basis {cx, h, s, rz}.
    2. Then applies Solovay-Kitaev synthesis to approximate Rz gates (and other
       single-qubit gates) into the target basis {cx, h, s, t, tdg}.

    Args:
        circuit (QuantumCircuit): The input quantum circuit.
        recursion_degree (int): The recursion degree for Solovay-Kitaev.
        remove_final_measurements (bool): If True, removes final measurements
                                          before transpilation. PBC usually redefines measurements.

    Returns:
        QuantumCircuit: The circuit transpiled to the Solovay-Kitaev basis.
    """
    processed_circuit = circuit.copy()

    if remove_final_measurements:
        processed_circuit.remove_final_measurements(inplace=True)

    # Step 1: Transpile to intermediate RZ basis
    print("Transpiling to intermediate RZ basis...")
    rz_circuit = transpile(
        processed_circuit, basis_gates=INTERMEDIATE_RZ_BASIS, optimization_level=0
    )
    print("Transpiled to intermediate RZ basis.")
    # Step 2: Apply Solovay-Kitaev synthesis
    # Build basic approximations with warnings suppressed (numpy.linalg may warn on det)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module=r".*numpy\.linalg.*"
        )
        old_err = np.seterr(divide="ignore", invalid="ignore")
        try:
            approx = generate_basic_approximations(
                basis_gates=SOLOVAY_KITAEV_BASIS, depth=5
            )
        finally:
            np.seterr(**old_err)
    sk_pass = SolovayKitaev(
        recursion_degree=recursion_degree, basic_approximations=approx
    )

    # Count gates that need SK synthesis
    gates_to_synthesize = [
        (i, item.operation, item.qubits)
        for i, item in enumerate(rz_circuit.data)
        if item.operation.name in ["rz", "u1", "u2", "u3", "u"]
    ]

    if gates_to_synthesize:
        print(f"      Found {len(gates_to_synthesize)} gates to synthesize...")

        # Apply SK synthesis (suppress noisy numpy.linalg RuntimeWarnings)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, module=r".*numpy\.linalg.*"
            )
            old_err = np.seterr(divide="ignore", invalid="ignore")
            try:
                discretized_circuit = sk_pass(rz_circuit)  # leaves sdg gates
            finally:
                np.seterr(**old_err)

        # Step 3: Final transpilation to target Clifford+T basis
        # discretized_circuit = transpile(
        #     discretized_circuit, basis_gates=SOLOVAY_KITAEV_BASIS + ["cx"], optimization_level=0
        # )  # Disable optimization for baseline
    else:
        print("      No gates requiring Solovay-Kitaev synthesis found")
        discretized_circuit = rz_circuit
        # discretized_circuit = transpile(
        #     rz_circuit, basis_gates=SOLOVAY_KITAEV_BASIS, optimization_level=0
        # )  # Disable optimization for baseline

    # Verify we're in the correct basis
    ops = discretized_circuit.count_ops()
    unexpected_gates = [
        gate
        for gate in ops
        if gate not in SOLOVAY_KITAEV_BASIS + ["cx", "barrier", "reset"]
    ]
    if unexpected_gates:
        print(
            f"      Warning: Unexpected gates found after SK synthesis: {unexpected_gates}"
        )

    if return_intermediate:
        return rz_circuit, discretized_circuit
    return discretized_circuit


def transpile_qasm_file_to_sk(
    input_qasm_path: str, output_qasm_path: str, recursion_degree: int
):
    """
    Helper to load QASM, transpile to Solovay-Kitaev basis, and save QASM.
    (Based on the original transpile_sk.py main block)
    """
    circuit = QuantumCircuit.from_qasm_file(input_qasm_path)
    discretized_circuit = transpile_to_solovay_kitaev_clifford_t(
        circuit, recursion_degree
    )
    with open(output_qasm_path, "w") as out_file:
        dump(discretized_circuit, out_file)
    return discretized_circuit
