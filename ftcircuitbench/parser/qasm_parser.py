# FTCircuitBench/ftcircuitbench/parser/qasm_parser.py
"""
Parser for QASM files and transpilation to a target basis set using Qiskit.
"""

import os
from typing import Optional

import qiskit.qasm3
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.transpiler import TranspilerError

# Qiskit gate names: 'rz', 's', 'h', 'x', 'cx'
DEFAULT_TARGET_BASIS = ["rz", "s", "h", "x", "cx"]

CLIFFORD_PLUS_RZ_BASIS = ["rz", "s", "h", "x", "z", "cx"]


def rebuild_circuit_with_single_qreg(circuit):
    # Get the quantum registers and their sizes
    total_qubits = sum(qreg.size for qreg in circuit.qregs)

    # Create a new quantum register that combines the qubits of all original registers
    new_qreg = QuantumRegister(total_qubits, "q")

    # Create new classical registers (assuming you want to keep all the original classical bits)
    new_cregs = []
    for creg in circuit.cregs:
        new_cregs.append(ClassicalRegister(creg.size, name=creg.name))

    # Create a new circuit with the combined quantum register and classical registers
    new_circuit = QuantumCircuit(new_qreg, *new_cregs)

    # Create a mapping from the old registers to the new register
    qubit_mapping = {}
    current_index = 0

    for qreg in circuit.qregs:
        for i in range(qreg.size):
            qubit_mapping[qreg[i]] = new_qreg[current_index]
            current_index += 1

    # Append all gates and measurements to the new circuit with remapped qubits
    for item in circuit.data:
        instruction, qargs, cargs = item.operation, item.qubits, item.clbits
        new_qargs = [qubit_mapping[qarg] for qarg in qargs]
        new_circuit.append(instruction, new_qargs, cargs)

    return new_circuit


def _detect_qasm_version(qasm_str: str) -> str:
    """
    Detect the OpenQASM version from the QASM string.

    Args:
        qasm_str (str): The QASM string to analyze.

    Returns:
        str: The detected version ('2.0' or '3.0'), defaults to '2.0' if not found.
    """
    lines = qasm_str.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("OPENQASM"):
            if "3.0" in line or "3" in line:
                return "3.0"
            elif "2.0" in line or "2" in line:
                return "2.0"
    # Default to 2.0 if no version statement found
    return "2.0"


def load_qasm_circuit(qasm_input: str, is_file: bool = True) -> QuantumCircuit:
    """
    Loads a quantum circuit from a QASM file or a QASM string.
    This serves as the first step, bringing the QASM into an internal
    representation (a Qiskit QuantumCircuit object).

    Supports both OpenQASM 2.0 and OpenQASM 3.0 formats.

    Args:
        qasm_input (str): Path to the QASM file or the QASM string itself.
        is_file (bool): True if qasm_input is a file path, False if it's a QASM string.
                        Defaults to True.

    Returns:
        QuantumCircuit: The loaded Qiskit QuantumCircuit object.

    Raises:
        FileNotFoundError: If qasm_input is a file path and the file does not exist.
        Exception: For other Qiskit-related parsing errors or general issues.
    """
    qasm_version = "2.0"
    filtered_qasm = ""
    try:
        # Read QASM content
        if is_file:
            if not os.path.exists(qasm_input):
                raise FileNotFoundError(f"QASM file not found: {qasm_input}")
            with open(qasm_input, "r") as f:
                qasm_str = f.read()
        else:
            qasm_str = qasm_input

        # Detect QASM version
        qasm_version = _detect_qasm_version(qasm_str)

        # Filter out reset operations at the QASM string level
        lines = qasm_str.split("\n")
        filtered_lines = []
        reset_count = 0

        for line in lines:
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith("//"):
                filtered_lines.append(line)
                continue

            # Check if line contains a reset operation
            if "reset" in line:
                reset_count += 1
                continue

            filtered_lines.append(line)

        if reset_count > 0:
            print(
                f"Warning: {reset_count} reset operations were found and removed from the QASM. "
                f"Reset operations are not supported in the PBC conversion process."
            )

        # Join the filtered lines and create the circuit
        filtered_qasm = "\n".join(filtered_lines)

        # Load circuit based on detected version
        if qasm_version == "3.0":
            circuit = qiskit.qasm3.loads(filtered_qasm)
        else:
            circuit = QuantumCircuit.from_qasm_str(filtered_qasm)

        new_circuit = rebuild_circuit_with_single_qreg(circuit)
        return new_circuit

    except FileNotFoundError:
        raise
    except Exception as e:
        # Try alternative loading method if the first one fails
        try:
            if qasm_version == "2.0":
                print("OpenQASM 2.0 loading failed, trying OpenQASM 3.0 loader...")
                if is_file:
                    circuit = qiskit.qasm3.load(qasm_input)
                else:
                    circuit = qiskit.qasm3.loads(filtered_qasm)
                new_circuit = rebuild_circuit_with_single_qreg(circuit)
                return new_circuit
            else:
                print("OpenQASM 3.0 loading failed, trying OpenQASM 2.0 loader...")
                circuit = QuantumCircuit.from_qasm_str(filtered_qasm)
                new_circuit = rebuild_circuit_with_single_qreg(circuit)
                return new_circuit
        except Exception as e2:
            # If both methods fail, raise the original error with additional context
            raise Exception(
                f"Error loading QASM (tried both OpenQASM 2.0 and 3.0): Original error: {e}, Alternative error: {e2}"
            )


def transpile_qasm_to_target_basis(
    qasm_input: str,
    is_file: bool = True,
    basis_gates: Optional[list] = None,
    optimization_level: int = 0,  # Disable optimization for baseline
) -> QuantumCircuit:
    # ... (content remains the same) ...
    if basis_gates is None:
        basis_gates = DEFAULT_TARGET_BASIS.copy()

    try:
        initial_circuit = load_qasm_circuit(qasm_input, is_file)
        transpiled_circuit = transpile(
            initial_circuit,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )
        return transpiled_circuit
    except FileNotFoundError:
        raise
    except TranspilerError as te:
        raise TranspilerError(
            f"Qiskit transpilation failed for basis {basis_gates}: {te}"
        )
    except Exception as e:
        raise Exception(
            f"An error occurred during QASM processing or transpilation: {e}"
        )
