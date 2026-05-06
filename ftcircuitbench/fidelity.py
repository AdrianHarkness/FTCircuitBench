"""
Fidelity calculation module for FTCircuitBench.
Provides scalable fidelity calculation methods for large quantum circuits.
"""

import multiprocessing
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.synthesis import generate_basic_approximations
from qiskit.transpiler.passes.synthesis import SolovayKitaev

from ftcircuitbench.decomposer import (
    _run_gridsynth_cli,
    create_circuit_from_gate_string,
)

# Configuration constants
# DEFAULT_GRIDSYNTH_PRECISION is kept for backward compatibility with older tests
DEFAULT_GRIDSYNTH_PRECISION = 5
MAX_QUBITS_FOR_FIDELITY = 7  # Maximum qubits for traditional unitary-based fidelity


def _calculate_single_rz_fidelity(
    args: Tuple[str, float, int],
) -> Tuple[str, float, float]:
    """
    Calculate fidelity for a single RZ gate decomposition.
    This function is designed to work with multiprocessing.

    Args:
        args: Tuple of (theta_str, theta_value, gridsynth_precision)

    Returns:
        Tuple of (theta_str, theta_value, fidelity)
    """
    theta_str, theta_value, gridsynth_precision = args

    try:
        # Create ideal RZ unitary
        ideal_rz_qc = QuantumCircuit(1)
        ideal_rz_qc.rz(theta_value, 0)
        ideal_rz_unitary = Operator(ideal_rz_qc)

        # Get gridsynth decomposition
        decomposed_sequence_str = _run_gridsynth_cli(
            theta_str, precision=gridsynth_precision
        )

        if not decomposed_sequence_str or all(
            g in "IW" for g in decomposed_sequence_str
        ):
            # Identity decomposition - perfect fidelity
            return theta_str, theta_value, 1.0

        # Create circuit from decomposition
        approx_qc = create_circuit_from_gate_string(decomposed_sequence_str)
        approx_unitary = Operator(approx_qc)

        # Calculate fidelity for this RZ gate
        fid = process_fidelity(
            approx_unitary, ideal_rz_unitary, require_cp=False, require_tp=False
        )

        return theta_str, theta_value, fid

    except Exception as e:
        # Re-raise the exception to expose the real error
        raise RuntimeError(
            f"Failed to calculate fidelity for RZ({theta_value}): {str(e)}"
        ) from e


def rz_product_fidelity(
    original_qc: QuantumCircuit,
    gridsynth_precision: int,
    use_multiprocessing: bool = True,
) -> Dict[str, Any]:
    """
    Calculate fidelity by tracking individual RZ gate decomposition fidelities.
    This method is more scalable for large circuits as it avoids computing full circuit unitaries.

    The method works by:
    1. Identifying all RZ gates in the circuit
    2. Decomposing each RZ gate using Gridsynth
    3. Calculating individual fidelity for each decomposition
    4. Computing overall fidelity as the product of individual fidelities

    This approach scales linearly with the number of RZ gates rather than exponentially with qubits.

    Args:
        original_qc: The original quantum circuit
        gridsynth_precision: Precision for gridsynth decomposition
        use_multiprocessing: Whether to use multiprocessing for parallel RZ decomposition

    Returns:
        dict: Contains overall fidelity, individual fidelities, and metadata
    """
    # Find all RZ gates in the circuit
    rz_gates = []
    for idx, item in enumerate(original_qc.data):
        op, qargs = item.operation, item.qubits
        if isinstance(op, RZGate):
            theta = op.params[0]
            qubit = qargs[0]

            # Convert angle to string for gridsynth
            if isinstance(theta, (int, float)):
                theta_str = f"{float(theta):.15g}"
            else:
                # Skip parameterized gates for now
                continue

            # Skip identity rotations
            if abs(float(theta_str)) < 1e-10:
                continue

            rz_gates.append((idx, qubit, theta_str, theta))

    if not rz_gates:
        return {
            "overall_fidelity": "N/A",
            "individual_fidelities": [],
            "rz_gate_count": 0,
            "status": "no_rz_gates",
            "method": "rz_product_fidelity",
        }

    # Calculate individual fidelities
    individual_fidelities = []
    failed_decompositions = 0

    if use_multiprocessing and len(rz_gates) > 1:
        # Use multiprocessing for parallel calculation
        try:
            with multiprocessing.Pool() as pool:
                # Prepare arguments for multiprocessing
                args_list = [
                    (theta_str, theta, gridsynth_precision)
                    for _, _, theta_str, theta in rz_gates
                ]

                # Calculate fidelities in parallel
                results = pool.map(_calculate_single_rz_fidelity, args_list)

                # Extract fidelities from results
                for theta_str, theta_value, fid in results:
                    individual_fidelities.append(fid)
                    if (
                        fid < 0.999
                    ):  # Threshold for considering it a "failed" decomposition
                        failed_decompositions += 1

        except Exception:
            # Fallback to sequential processing if multiprocessing fails
            use_multiprocessing = False

    if not use_multiprocessing or len(rz_gates) <= 1:
        # Sequential processing
        for gate_idx, (idx, qubit, theta_str, theta) in enumerate(rz_gates):
            try:
                # Create ideal RZ unitary
                ideal_rz_qc = QuantumCircuit(1)
                ideal_rz_qc.rz(theta, 0)
                ideal_rz_unitary = Operator(ideal_rz_qc)

                # Get gridsynth decomposition
                decomposed_sequence_str = _run_gridsynth_cli(
                    theta_str, precision=gridsynth_precision
                )

                if not decomposed_sequence_str or all(
                    g in "IW" for g in decomposed_sequence_str
                ):
                    individual_fidelities.append(1.0)  # Identity decomposition
                    continue

                # Create circuit from decomposition
                approx_qc = create_circuit_from_gate_string(decomposed_sequence_str)
                approx_unitary = Operator(approx_qc)

                # Calculate fidelity for this RZ gate
                fid = process_fidelity(
                    approx_unitary, ideal_rz_unitary, require_cp=False, require_tp=False
                )
                individual_fidelities.append(fid)

            except Exception as e:
                # Re-raise the exception to expose the real error
                raise RuntimeError(
                    f"Failed to calculate fidelity for RZ gate {gate_idx}: {str(e)}"
                ) from e

    # Calculate overall fidelity as product of individual fidelities
    overall_fidelity = np.prod(individual_fidelities)

    result = {
        "overall_fidelity": overall_fidelity,
        "individual_fidelities": individual_fidelities,
        "rz_gate_count": len(rz_gates),
        "failed_decompositions": failed_decompositions,
        "min_individual_fidelity": (
            min(individual_fidelities) if individual_fidelities else 1.0
        ),
        "max_individual_fidelity": (
            max(individual_fidelities) if individual_fidelities else 1.0
        ),
        "avg_individual_fidelity": (
            np.mean(individual_fidelities) if individual_fidelities else 1.0
        ),
        "status": "success" if failed_decompositions == 0 else "partial_failure",
        "method": "rz_product_fidelity",
        "multiprocessing_used": use_multiprocessing and len(rz_gates) > 1,
    }

    return result


def _synthesize_single_rz_with_sk(
    theta_value: float, recursion_degree: int
) -> QuantumCircuit:
    """
    Synthesize a single-qubit RZ(theta) using Solovay-Kitaev and return the approximating circuit.

    Args:
        theta_value: The rotation angle for RZ.
        recursion_degree: Recursion degree for SK synthesis.

    Returns:
        QuantumCircuit: Approximated single-qubit circuit in {h, s, t, tdg} basis.
    """
    # Ideal RZ circuit (1 qubit)
    src = QuantumCircuit(1)
    src.rz(theta_value, 0)

    # Build SK approximations library
    approx = generate_basic_approximations(basis_gates=["h", "s", "t", "tdg"], depth=5)
    sk = SolovayKitaev(recursion_degree=recursion_degree, basic_approximations=approx)

    # Apply SK synthesis pass to approximate the single-qubit unitary
    approx_qc = sk(src)
    return approx_qc


def rz_product_fidelity_sk(
    intermediate_rz_qc: QuantumCircuit,
    recursion_degree: int,
    use_multiprocessing: bool = True,
) -> Dict[str, Any]:
    """
    Calculate fidelity by approximating each RZ gate using Solovay-Kitaev and multiplying
    individual fidelities, analogous to the Gridsynth-based rz_product_fidelity but without
    calling Gridsynth.

    Args:
        intermediate_rz_qc: Circuit expressed in an intermediate {rz, h, s, cx} basis.
        recursion_degree: Recursion degree to use for Solovay-Kitaev per-gate synthesis.
        use_multiprocessing: Whether to parallelize per-RZ synthesis.

    Returns:
        dict with overall_fidelity, individual_fidelities, rz_gate_count, etc.
    """
    # Collect all concrete RZ gates
    rz_thetas: List[float] = []
    for item in intermediate_rz_qc.data:
        op = item.operation
        if isinstance(op, RZGate):
            theta = op.params[0]
            try:
                theta_f = float(theta)
            except Exception:
                # Skip parameterized gates for now
                continue
            if abs(theta_f) < 1e-10:
                continue
            rz_thetas.append(theta_f)

    if not rz_thetas:
        return {
            "overall_fidelity": "N/A",
            "individual_fidelities": [],
            "rz_gate_count": 0,
            "status": "no_rz_gates",
            "method": "rz_product_fidelity_sk",
        }

    def _fid_for_theta(theta_value: float) -> float:
        ideal = QuantumCircuit(1)
        ideal.rz(theta_value, 0)
        ideal_u = Operator(ideal)

        approx_qc = _synthesize_single_rz_with_sk(theta_value, recursion_degree)
        approx_u = Operator(approx_qc)
        return float(
            process_fidelity(approx_u, ideal_u, require_cp=False, require_tp=False)
        )

    individual_fidelities: List[float] = []
    if use_multiprocessing and len(rz_thetas) > 1:
        try:
            with multiprocessing.Pool() as pool:
                individual_fidelities = pool.map(_fid_for_theta, rz_thetas)
        except Exception:
            # Fallback to sequential if multiprocessing fails
            individual_fidelities = [_fid_for_theta(theta) for theta in rz_thetas]
    else:
        individual_fidelities = [_fid_for_theta(theta) for theta in rz_thetas]

    overall_fidelity = (
        float(np.prod(individual_fidelities)) if individual_fidelities else 1.0
    )
    return {
        "overall_fidelity": overall_fidelity,
        "individual_fidelities": individual_fidelities,
        "rz_gate_count": len(rz_thetas),
        "failed_decompositions": 0,
        "min_individual_fidelity": (
            min(individual_fidelities) if individual_fidelities else 1.0
        ),
        "max_individual_fidelity": (
            max(individual_fidelities) if individual_fidelities else 1.0
        ),
        "avg_individual_fidelity": (
            np.mean(individual_fidelities) if individual_fidelities else 1.0
        ),
        "status": "success",
        "method": "rz_product_fidelity_sk",
        "multiprocessing_used": use_multiprocessing and len(rz_thetas) > 1,
    }


def calculate_circuit_fidelity(
    original_qc: QuantumCircuit,
    decomposed_qc: QuantumCircuit,
    gridsynth_precision: int,
    sk_recursion_degree: Optional[int] = None,
    intermediate_qc: Optional[QuantumCircuit] = None,
) -> Dict[str, Any]:
    """
    Calculate fidelity between original and decomposed circuits.
    For large circuits (> MAX_QUBITS_FOR_FIDELITY), uses rz_product_fidelity.
    For smaller circuits, uses traditional unitary-based fidelity.

    This function automatically handles custom gates by using the intermediate Clifford+RZ
    representation when available. If intermediate_qc is not provided, it will attempt
    to use the original circuit (which may not work for circuits with custom gates).

    Args:
        original_qc: The original quantum circuit (may contain custom gates)
        decomposed_qc: The decomposed quantum circuit (Clifford+T)
        gridsynth_precision: Precision for gridsynth decomposition (used for rz_product_fidelity)
        intermediate_qc: Optional intermediate Clifford+RZ circuit (recommended for circuits with custom gates)

    Returns:
        dict: Contains fidelity value, method used, and status
    """
    if original_qc.num_qubits <= MAX_QUBITS_FOR_FIDELITY:
        # Use traditional unitary-based fidelity for small circuits
        try:
            # Remove measurements if present for unitary-based calculation
            original_qc_clean = original_qc.copy()
            original_qc_clean.remove_final_measurements(inplace=True)
            decomposed_qc_clean = decomposed_qc.copy()
            decomposed_qc_clean.remove_final_measurements(inplace=True)

            original_unitary = Operator(original_qc_clean)
            decomposed_unitary = Operator(decomposed_qc_clean)
            fidelity = process_fidelity(decomposed_unitary, original_unitary)

            return {
                "fidelity": fidelity,
                "method": "unitary_based",
                "status": "success",
            }
        except Exception as e:
            return {
                "fidelity": None,
                "method": "unitary_based",
                "status": f"error: {str(e)}",
            }
    else:
        # Use rz_product_fidelity for large circuits
        try:
            # Require an intermediate circuit for scalable fidelity
            if intermediate_qc is None:
                # For Solovay-Kitaev or other pipelines without intermediate circuit,
                # we cannot accurately calculate fidelity for large circuits with custom gates
                return {
                    "fidelity": "N/A",
                    "method": (
                        "rz_product_fidelity_sk"
                        if sk_recursion_degree is not None
                        else "rz_product_fidelity"
                    ),
                    "status": "not_available_no_intermediate_circuit",
                    "rz_gate_count": 0,
                    "individual_fidelities": [],
                }

            # Use the intermediate circuit for accurate RZ product fidelity calculation
            if sk_recursion_degree is not None:
                rz_fid_result = rz_product_fidelity_sk(
                    intermediate_qc, sk_recursion_degree
                )
                return {
                    "fidelity": rz_fid_result["overall_fidelity"],
                    "method": "rz_product_fidelity_sk",
                    "status": rz_fid_result["status"],
                    "rz_gate_count": rz_fid_result["rz_gate_count"],
                    "individual_fidelities": rz_fid_result["individual_fidelities"],
                }
            else:
                rz_fid_result = rz_product_fidelity(
                    intermediate_qc, gridsynth_precision
                )
                return {
                    "fidelity": rz_fid_result["overall_fidelity"],
                    "method": "rz_product_fidelity",
                    "status": rz_fid_result["status"],
                    "rz_gate_count": rz_fid_result["rz_gate_count"],
                    "individual_fidelities": rz_fid_result["individual_fidelities"],
                }
        except RuntimeError as e:
            # Handle specific runtime errors (like gridsynth not available)
            if "gridsynth" in str(e).lower() or "command" in str(e).lower():
                return {
                    "fidelity": "N/A",
                    "method": "rz_product_fidelity",
                    "status": "gridsynth_not_available",
                    "error_message": str(e),
                    "rz_gate_count": 0,
                    "individual_fidelities": [],
                }
            else:
                return {
                    "fidelity": "N/A",
                    "method": "rz_product_fidelity",
                    "status": "calculation_failed",
                    "error_message": str(e),
                }
        except Exception as e:
            return {
                "fidelity": "N/A",
                "method": "rz_product_fidelity",
                "status": "unexpected_error",
                "error_message": str(e),
            }
