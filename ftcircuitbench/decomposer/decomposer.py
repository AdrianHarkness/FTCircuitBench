# ./ftcircuitbench/decomposer/decomposer.py
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.circuit.library import RZGate

# For __main__ fidelity check:

# --- Gridsynth CLI Based Decomposition ---


def _run_gridsynth_cli(angle: Union[float, str], precision: int = 10) -> str:
    """
    Decompose an angle into a series of S, H, T gates using the gridsynth CLI.

    Gridsynth must be installed and accessible in the system's PATH.
    It can parse numerical values and symbolic expressions like "pi/4".
    Scientific notation will be converted to decimal format.

    Note: This function is primarily used with string inputs in the codebase.
    Float inputs are supported mainly for testing purposes.

    :param angle: Angle in radians as a float or string (e.g., 0.785398, "pi/4").
                 String inputs are preferred, especially for symbolic expressions.
    :param precision: Number of digits of precision for gridsynth.
    :return: String of the gate sequence (e.g., "THTHTHS").
             The order of gates in the string is intended for direct application
             from left to right to construct the desired rotation.
    :raises RuntimeError: If gridsynth command fails, is not found, or returns an error.
    """
    try:
        # Convert float to string if needed (mainly for testing)
        if isinstance(angle, float):
            # Convert to decimal format with sufficient precision
            angle_str = f"{angle:.20f}"
        else:
            # For string inputs, only convert numeric strings to decimal format
            # Leave symbolic expressions (like "pi/4") as is
            if isinstance(angle, str) and "pi" not in angle.lower():
                try:
                    # Convert to float and then to decimal format
                    angle_str = f"{float(angle):.20f}"
                except ValueError:
                    # If conversion fails, use the string as is
                    angle_str = angle
            else:
                angle_str = angle

        # Construct Gridsynth command with parentheses around the angle
        # This prevents negative values from being interpreted as command line arguments
        cmd = f'gridsynth "({angle_str})" --digits={precision}'

        # Run Gridsynth
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        error_message = e.stderr.strip() if e.stderr else "Unknown error"
        raise RuntimeError(f"gridsynth command failed with error: {error_message}")
    except (ValueError, SyntaxError) as e:
        raise RuntimeError(f"Could not parse angle value '{angle}'. Error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error running Gridsynth: {str(e)}")


def _run_gridsynth_cli_unpack(args):
    return _run_gridsynth_cli(*args)


def decompose_rz_gates_gridsynth(
    original_circuit: QuantumCircuit,
    precision: int = 10,
    progress_bar=None,
    return_decomp_map: bool = False,
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, Dict[str, str]]]:
    """
    Decomposes all RZ gates in a quantum circuit into S, H, T (and possibly X) gates
    using the gridsynth CLI.

    Note:
    - Gridsynth must be installed and in the system PATH.
    - Circuit parameters in RZ gates must be bound to numerical values or
      expressions that gridsynth can parse (e.g., "pi/4"). Unbound Qiskit
      `ParameterExpression` objects (e.g., containing `Parameter('my_angle')`)
      will cause an error.
    - Identity operations (rotations of 0 degrees) are automatically removed.
    - Gridsynth is invoked at most once per unique angle string; repeats are
      cached and reused.

    Args:
        original_circuit: The Qiskit QuantumCircuit to decompose.
        precision: Number of digits of precision for gridsynth.
        progress_bar: Optional tqdm progress bar to update during decomposition.
        return_decomp_map: If True, also return a dict mapping each unique
            theta string to its gridsynth gate-string output. Useful for
            downstream consumers (e.g. fidelity calculation) that would
            otherwise re-invoke gridsynth on the same angles.

    Returns:
        A new QuantumCircuit with RZ gates replaced by their decompositions.
        If return_decomp_map=True, returns (new_circuit, decomp_map) instead.

    Raises:
        RuntimeError: If gridsynth command fails or is not found.
        ValueError: If RZ gate parameters are unsuitable for gridsynth.
    """
    new_circuit = QuantumCircuit(
        *original_circuit.qregs,
        *original_circuit.cregs,
        name=(
            original_circuit.name + "_decomposed_rz"
            if original_circuit.name
            else "decomposed_rz"
        ),
    )

    ZERO_THRESHOLD = 10 ** (-precision)
    # Step 1: Collect all operations and RZ gate info
    ops_info: List[Tuple[bool, Any, Any, Any, Any]] = (
        []
    )  # (is_rz, op, qargs, cargs, rz_info)
    rz_jobs: List[Tuple[int, Any, str]] = []  # (index, qubit, theta_str)
    for idx, instr in enumerate(original_circuit.data):
        op, qargs, cargs = instr.operation, instr.qubits, instr.clbits
        if isinstance(op, RZGate):
            theta = op.params[0]
            qubit = qargs[0]
            # Convert angle to string
            if isinstance(theta, (int, float)):
                theta_str = f"{float(theta):.15g}"
            elif isinstance(theta, ParameterExpression):
                try:
                    theta_val = float(theta)
                    theta_str = f"{theta_val:.15g}"
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"RZ gate parameter {theta} is a ParameterExpression that "
                        f"could not be evaluated to a number. Gridsynth requires "
                        f"numerical values. Error: {e}"
                    )
            else:
                raise ValueError(
                    f"RZ gate parameter {theta} is of unsupported type {type(theta)}. "
                    f"Gridsynth requires numerical values."
                )
            # Skip identity
            if abs(float(theta_str)) < ZERO_THRESHOLD:
                if progress_bar:
                    progress_bar.update(1)
                ops_info.append((False, None, None, None, None))
                continue
            rz_jobs.append((idx, qubit, theta_str))
            ops_info.append((True, None, None, None, (qubit, theta_str)))
        else:
            ops_info.append((False, op, qargs, cargs, None))

    # Step 2: Process RZ gates (dedupe identical theta strings)
    rz_results: Dict[int, Tuple[Any, str]] = {}
    decomp_map: Dict[str, str] = {}
    if rz_jobs:
        for idx, qubit, theta_str in rz_jobs:
            try:
                if theta_str not in decomp_map:
                    decomp_map[theta_str] = _run_gridsynth_cli(theta_str, precision)
                rz_results[idx] = (qubit, decomp_map[theta_str])
                if progress_bar:
                    progress_bar.update(1)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to decompose RZ gate at index {idx} with angle {theta_str}: {str(e)}"
                )

    # Step 3: Reconstruct circuit in original order
    for idx, (is_rz, op, qargs, cargs, rz_info) in enumerate(ops_info):
        if is_rz:
            qubit, theta_str = rz_info
            decomp_str = rz_results.get(idx, (None, ""))[1]
            # Skip if empty or only identity
            if not decomp_str or all(g in "IW" for g in decomp_str):
                if progress_bar:
                    progress_bar.update(1)
                continue
            for gate_char in decomp_str:
                if gate_char == "S":
                    new_circuit.s(qubit)
                elif gate_char == "T":
                    new_circuit.t(qubit)
                elif gate_char == "H":
                    new_circuit.h(qubit)
                elif gate_char == "X":
                    new_circuit.x(qubit)
                elif gate_char in "IW":
                    continue
                else:
                    print(
                        f"Warning: Encountered unexpected gate character '{gate_char}' "
                        f"in gridsynth output for Rz({theta_str}). This gate will be ignored."
                    )
            if progress_bar:
                progress_bar.update(1)
        elif op is not None:
            new_circuit.append(op, qargs, cargs)
    if return_decomp_map:
        return new_circuit, decomp_map
    return new_circuit


def create_circuit_from_gate_string(gate_sequence: str) -> QuantumCircuit:
    """
    Generates a 1-qubit Qiskit circuit from a sequence of gate characters.
    Operators are applied from left to right as they appear in the string.
    e.g., "STH" will apply S, then T, then H. The resulting unitary is U_H U_T U_S.

    :param gate_sequence: String of gate characters (e.g., "SHTH").
                          Supported characters: S, T, H, X, Z, Y.
                          Case-insensitive for input characters.
    :return: A QuantumCircuit object with one qubit.
    :raises TypeError: if gate_sequence is not a string.
    :raises ValueError: if gate_sequence contains unsupported characters.
    """
    if not isinstance(gate_sequence, str):
        raise TypeError("gate_sequence must be a string.")

    qc = QuantumCircuit(1, name=f"seq_{gate_sequence[:10]}")

    gate_methods = {
        "S": qc.s,
        "T": qc.t,
        "H": qc.h,
        "X": qc.x,
        "Z": qc.z,
        "Y": qc.y,
        "W": qc.id,
    }

    for gate_char_orig in gate_sequence:
        gate_char = gate_char_orig.upper()
        method = gate_methods.get(gate_char)
        if method:
            method(0)
        else:
            raise ValueError(
                f"Unsupported gate character '{gate_char_orig}' in sequence. "
                f"Supported characters are: {', '.join(gate_methods.keys())}."
            )
    return qc


def parse_angle_from_gate_name(gate_name: str) -> Optional[float]:
    """
    Extracts and evaluates an angle from a gate name string like "rz(angle_expression)".
    Example: "rz(0.785)", "rz(Pi/4)", "rz(numpy.pi/2)".
    Uses `eval()` safely.
    """
    if not isinstance(gate_name, str):
        raise TypeError("gate_name must be a string")

    match = re.search(r"\(([^)]+)\)", gate_name)
    if match:
        angle_str = match.group(1)
        try:
            allowed_globals: Dict[str, Any] = {"__builtins__": {}}
            allowed_locals = {
                "pi": np.pi,
                "Pi": np.pi,
                "numpy": np,
                "np": np,
                "sqrt": np.sqrt,
                "cos": np.cos,
                "sin": np.sin,
                "exp": np.exp,
            }
            angle = eval(angle_str, allowed_globals, allowed_locals)
            return float(angle)
        except Exception:
            return None
    return None


# --- PyGridsynth Based Decomposition (Placeholders) ---


def _pygridsynth_decompose_angle(angle: float, precision_digits: int = 10) -> str:
    raise NotImplementedError(
        "PyGridsynth _pygridsynth_decompose_angle is not yet implemented."
    )


def decompose_rz_gates_pygridsynth(
    original_circuit: QuantumCircuit, precision_digits: int = 10
) -> QuantumCircuit:
    raise NotImplementedError(
        "PyGridsynth RZ gate decomposition (decompose_rz_gates_pygridsynth) "
        "is not yet implemented."
    )
