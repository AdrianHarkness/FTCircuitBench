# ./ftcircuitbench/transpilers/sk_transpiler.py
"""
Solovay-Kitaev transpilation to Clifford+T basis.

Pipeline shape is the canonical one defined in `_basis.py`. Only the RZ-synthesis
step is engine-specific (Solovay-Kitaev approximation here).
"""

import warnings
from typing import Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dump
from qiskit.synthesis import generate_basic_approximations
from qiskit.transpiler.passes.synthesis import SolovayKitaev

from ._basis import (
    INTERMEDIATE_RZ_BASIS,
    enforce_pbc_basis,
    is_clifford_t_basis,
    prepare_input,
    to_intermediate_rz,
)

SOLOVAY_KITAEV_BASIS = ["h", "s", "sdg", "t", "tdg", "x", "y", "z"]


def transpile_to_solovay_kitaev_clifford_t(
    circuit: Union[QuantumCircuit, str],
    recursion_degree: int = 3,
    remove_final_measurements: bool = True,
    return_intermediate: bool = False,
    is_file: bool = False,
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, QuantumCircuit]]:
    """Transpile to Clifford+T via Solovay-Kitaev synthesis.

    Pipeline:
      1. prepare_input (QC/QASM, optional measurement removal)
      2. early-return if already in PBC Clifford+T basis
      3. to_intermediate_rz (canonical {cx, h, s, rz} basis)
      4. Solovay-Kitaev synthesis of remaining single-qubit rotations
      5. enforce_pbc_basis ({cx, h, s, t, tdg})  (cleans up sdg etc.)
    """
    processed_circuit = prepare_input(
        circuit, is_file=is_file, remove_final_measurements=remove_final_measurements
    )

    if is_clifford_t_basis(processed_circuit):
        print(
            "      Circuit is already in Clifford+T basis. Skipping SK transpilation."
        )
        return (
            (processed_circuit, processed_circuit)
            if return_intermediate
            else processed_circuit
        )

    print("Transpiling to intermediate RZ basis...")
    rz_circuit = to_intermediate_rz(processed_circuit)
    print("Transpiled to intermediate RZ basis.")

    # Solovay-Kitaev approximation library setup; numpy.linalg can warn benignly.
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

    gates_to_synthesize = [
        (i, item.operation, item.qubits)
        for i, item in enumerate(rz_circuit.data)
        if item.operation.name in ["rz", "u1", "u2", "u3", "u"]
    ]

    if gates_to_synthesize:
        print(f"      Found {len(gates_to_synthesize)} gates to synthesize...")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, module=r".*numpy\.linalg.*"
            )
            old_err = np.seterr(divide="ignore", invalid="ignore")
            try:
                discretized_circuit = sk_pass(rz_circuit)
            finally:
                np.seterr(**old_err)
    else:
        print("      No gates requiring Solovay-Kitaev synthesis found")
        discretized_circuit = rz_circuit

    discretized_circuit = enforce_pbc_basis(discretized_circuit)
    return (
        (rz_circuit, discretized_circuit)
        if return_intermediate
        else discretized_circuit
    )


def transpile_qasm_file_to_sk(
    input_qasm_path: str, output_qasm_path: str, recursion_degree: int
):
    """Helper: load QASM, transpile via Solovay-Kitaev, dump QASM."""
    circuit = QuantumCircuit.from_qasm_file(input_qasm_path)
    discretized_circuit = transpile_to_solovay_kitaev_clifford_t(
        circuit, recursion_degree
    )
    with open(output_qasm_path, "w") as out_file:
        dump(discretized_circuit, out_file)
    return discretized_circuit


__all__ = [
    "INTERMEDIATE_RZ_BASIS",
    "SOLOVAY_KITAEV_BASIS",
    "transpile_to_solovay_kitaev_clifford_t",
    "transpile_qasm_file_to_sk",
]
