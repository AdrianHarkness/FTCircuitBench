"""
Fidelity calculation module for FTCircuitBench.
Provides scalable fidelity calculation methods for large quantum circuits.
"""

import functools
import multiprocessing
import warnings
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
# Per-gate fidelity at or above this is considered a "good" decomposition.
FAILED_DECOMPOSITION_THRESHOLD = 0.999


def _unitary_process_fidelity_1q(approx_op: Operator, ideal_op: Operator) -> float:
    """
    Closed-form process (entanglement) fidelity for two single-qubit unitaries:
    F = |tr(U_approx^dagger U_ideal)|^2 / d^2 with d=2.

    Numerically equivalent to qiskit.quantum_info.process_fidelity(U, V,
    require_cp=False, require_tp=False) for unitary Operators, but avoids the
    Choi-state construction qiskit performs internally.
    """
    # np.vdot(a, b) computes sum_ij conj(a[i,j]) * b[i,j] = tr(a^dagger b).
    trace = np.vdot(approx_op.data, ideal_op.data)
    return float((trace * trace.conjugate()).real / 4.0)


def _calculate_single_rz_fidelity(
    args: Tuple[str, float, int, Optional[str]],
) -> Tuple[str, float, float]:
    """
    Calculate fidelity for a single RZ gate decomposition.
    This function is designed to work with multiprocessing.

    Args:
        args: Tuple of (theta_str, theta_value, gridsynth_precision,
              precomputed_decomp). When precomputed_decomp is not None it is
              used directly and gridsynth is not invoked.

    Returns:
        Tuple of (theta_str, theta_value, fidelity)
    """
    theta_str, theta_value, gridsynth_precision, precomputed_decomp = args

    try:
        # Create ideal RZ unitary
        ideal_rz_qc = QuantumCircuit(1)
        ideal_rz_qc.rz(theta_value, 0)
        ideal_rz_unitary = Operator(ideal_rz_qc)

        if precomputed_decomp is not None:
            decomposed_sequence_str = precomputed_decomp
        else:
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

        fid = _unitary_process_fidelity_1q(approx_unitary, ideal_rz_unitary)

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
    decomp_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Calculate fidelity by tracking individual RZ gate decomposition fidelities.
    This method is more scalable for large circuits as it avoids computing full circuit unitaries.

    The method works by:
    1. Identifying all RZ gates in the circuit
    2. Decomposing each RZ gate using Gridsynth (or reusing a precomputed decomposition)
    3. Calculating individual fidelity for each decomposition
    4. Computing overall fidelity as the product of individual fidelities

    This approach scales linearly with the number of RZ gates rather than exponentially with qubits.

    If a decomposition map is supplied (either via the ``decomp_map`` argument or
    on ``original_qc.metadata["gridsynth_decomp"]``), gridsynth is not invoked
    again for any angle present in the map — the cached gate string is reused.

    Args:
        original_qc: The original quantum circuit
        gridsynth_precision: Precision for gridsynth decomposition
        use_multiprocessing: Whether to use multiprocessing for parallel RZ decomposition
        decomp_map: Optional ``{theta_str -> gate_string}`` from a prior
            gridsynth run; when provided, avoids redundant subprocess calls.

    Returns:
        dict: Contains overall fidelity, individual fidelities, and metadata
    """
    if decomp_map is None:
        md = getattr(original_qc, "metadata", None)
        if isinstance(md, dict):
            decomp_map = md.get("gridsynth_decomp")
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

    # Per-gate fidelity is a pure function of theta_str (and the gridsynth
    # decomposition it implies), so we compute once per *unique* angle and
    # fan the result back out to every occurrence.
    unique_args: Dict[str, Tuple[str, float, int, Optional[str]]] = {}
    for _, _, theta_str, theta in rz_gates:
        if theta_str not in unique_args:
            unique_args[theta_str] = (
                theta_str,
                theta,
                gridsynth_precision,
                decomp_map.get(theta_str) if decomp_map else None,
            )
    unique_arg_list = list(unique_args.values())

    unique_fids: Dict[str, float] = {}
    if use_multiprocessing and len(unique_arg_list) > 1:
        try:
            with multiprocessing.Pool() as pool:
                results = pool.map(_calculate_single_rz_fidelity, unique_arg_list)
                for _ts, _theta_value, fid in results:
                    unique_fids[_ts] = fid
        except Exception as e:
            warnings.warn(
                f"rz_product_fidelity multiprocessing failed ({e!r}); "
                f"falling back to sequential processing.",
                RuntimeWarning,
                stacklevel=2,
            )
            use_multiprocessing = False
            unique_fids = {}

    if not use_multiprocessing or len(unique_arg_list) <= 1:
        for args in unique_arg_list:
            _ts, _theta_value, fid = _calculate_single_rz_fidelity(args)
            unique_fids[_ts] = fid

    individual_fidelities: List[float] = [
        unique_fids[theta_str] for _, _, theta_str, _ in rz_gates
    ]

    # Calculate overall fidelity as product of individual fidelities
    overall_fidelity = np.prod(individual_fidelities)
    failed_decompositions = sum(
        1 for f in individual_fidelities if f < FAILED_DECOMPOSITION_THRESHOLD
    )

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
        "multiprocessing_used": use_multiprocessing and len(unique_arg_list) > 1,
        "unique_angle_count": len(unique_arg_list),
    }

    return result


@functools.lru_cache(maxsize=None)
def _sk_basic_approximations(
    basis_gates: Tuple[str, ...] = ("h", "s", "t", "tdg"), depth: int = 5
):
    # Cache the basis-approximation library: it depends only on (basis_gates, depth)
    # and is by far the most expensive part of an SK pass.
    return generate_basic_approximations(basis_gates=list(basis_gates), depth=depth)


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

    sk = SolovayKitaev(
        recursion_degree=recursion_degree,
        basic_approximations=_sk_basic_approximations(),
    )

    # Apply SK synthesis pass to approximate the single-qubit unitary
    approx_qc = sk(src)
    return approx_qc


def _sk_fidelity_for_theta(theta_value: float, recursion_degree: int) -> float:
    """Single-Rz SK fidelity. Module-level so it's picklable for multiprocessing."""
    ideal = QuantumCircuit(1)
    ideal.rz(theta_value, 0)
    ideal_u = Operator(ideal)

    approx_qc = _synthesize_single_rz_with_sk(theta_value, recursion_degree)
    approx_u = Operator(approx_qc)
    return _unitary_process_fidelity_1q(approx_u, ideal_u)


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

    # Per-theta fidelity is a pure function of the angle, so we synthesize
    # once per unique theta and fan the result out to every occurrence.
    unique_thetas = list(dict.fromkeys(rz_thetas))

    unique_fids: Dict[float, float] = {}
    if use_multiprocessing and len(unique_thetas) > 1:
        try:
            args = [(t, recursion_degree) for t in unique_thetas]
            with multiprocessing.Pool() as pool:
                for theta, fid in zip(
                    unique_thetas, pool.starmap(_sk_fidelity_for_theta, args)
                ):
                    unique_fids[theta] = fid
        except Exception as e:
            warnings.warn(
                f"rz_product_fidelity_sk multiprocessing failed ({e!r}); "
                f"falling back to sequential processing.",
                RuntimeWarning,
                stacklevel=2,
            )
            use_multiprocessing = False
            unique_fids = {
                t: _sk_fidelity_for_theta(t, recursion_degree) for t in unique_thetas
            }
    else:
        unique_fids = {
            t: _sk_fidelity_for_theta(t, recursion_degree) for t in unique_thetas
        }

    individual_fidelities: List[float] = [unique_fids[t] for t in rz_thetas]

    overall_fidelity = (
        float(np.prod(individual_fidelities)) if individual_fidelities else 1.0
    )
    failed_decompositions = sum(
        1 for f in individual_fidelities if f < FAILED_DECOMPOSITION_THRESHOLD
    )
    return {
        "overall_fidelity": overall_fidelity,
        "individual_fidelities": individual_fidelities,
        "rz_gate_count": len(rz_thetas),
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
        "method": "rz_product_fidelity_sk",
        "multiprocessing_used": use_multiprocessing and len(unique_thetas) > 1,
        "unique_angle_count": len(unique_thetas),
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
