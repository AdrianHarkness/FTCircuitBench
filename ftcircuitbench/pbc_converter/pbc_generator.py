# ftcircuitbench/pbc_converter/pbc_generator.py
"""
Generates Pauli Based Computation (PBC) circuits from Clifford+T circuits.
This file contains the unified logic for PBC conversion using parallel RPC fallback.
"""

import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from ftcircuitbench.pbc_converter.nwqec_adapter import (
    is_nwqec_available,
    transpile_to_pbc_cpp,
)

from .pbc_circuit_saver import save_pbc_layers_txt, save_pbc_measurement_basis_txt
from .pbm import PBM, Rotation
from .r_pauli_circ import RotationPauliCirc
from .tab_gate import TableauForGate, TableauPauliBasis

# --- Parallel Helper Functions ---


def parallel_t_merging_optimized(
    t_layers: List[TableauForGate],
    measure_tab: TableauForGate,
    max_workers: Optional[int] = None,
) -> Tuple[List[TableauForGate], TableauForGate]:
    """
    Parallelized version of T-gate merging with a two-phase approach.
    This is a helper function used by ParallelRotationPauliCirc.
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 12)
    if not t_layers:
        return [], measure_tab

    def simplify_layer(tab: TableauForGate) -> Tuple[TableauPauliBasis, List, List]:
        layer = TableauPauliBasis(tab.tableau.copy())
        z_gates, s_gates = layer.simplify()
        return layer, z_gates, s_gates

    actual_workers = min(max_workers, len(t_layers))
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        results = list(executor.map(simplify_layer, t_layers))

    simplified_layers = [r[0] for r in results]
    clifford_gates_by_layer = [(r[1], r[2]) for r in results]

    measure_tab_copy = measure_tab.tableau.copy()
    for j in range(len(simplified_layers) - 1, -1, -1):
        z_gates, s_gates = clifford_gates_by_layer[j]
        if not z_gates and not s_gates:
            continue

        for k in range(j + 1, len(simplified_layers)):
            for z in z_gates:
                simplified_layers[k].commute_pauli(z)
            for s in s_gates:
                simplified_layers[k].front_multiply_pauli(s)

        measure_tab_obj = TableauForGate(measure_tab_copy)
        for z in z_gates:
            measure_tab_obj.commute_pauli(z)
        for s in s_gates:
            measure_tab_obj.front_multiply_pauli(s)
        measure_tab_copy = measure_tab_obj.tableau

    optimized_t_layers = simplified_layers
    updated_measure_tab = TableauForGate(measure_tab_copy)
    return optimized_t_layers, updated_measure_tab


# --- ParallelRotationPauliCirc Class ---


class ParallelRotationPauliCirc(RotationPauliCirc):
    """
    Parallelized version of RotationPauliCirc with optimized performance.
    Overrides t_merging method to use parallel helpers.
    """

    def __init__(self, qc: QuantumCircuit, max_workers: Optional[int] = None):
        super().__init__(qc)
        self.max_workers = max_workers or min(mp.cpu_count(), 12)

    def t_merging(self, debug=False):
        """Overridden T-merging method to use the optimized parallel implementation."""
        if self.t_layers is None:
            self.layering()

        if self.t_layers:
            self.t_layers, self.measure_tab = parallel_t_merging_optimized(
                self.t_layers, self.measure_tab, max_workers=self.max_workers
            )


# --- Internal Unified Analysis and Formatting ---


def _format_pbc_optimization_results(
    rpc: RotationPauliCirc,
    initial_clifford_t_t_count: int,
    gate_ct_tracker: list,
    rpc_stats_list: list,
) -> dict:
    """Internal helper to format the statistics from an RPC object."""
    circuit_stats = {}
    circuit_stats["num_qubits"] = rpc.num_qubits
    circuit_stats["initial_clifford_t_t_gates_for_pbc"] = initial_clifford_t_t_count

    if gate_ct_tracker:
        circuit_stats["processed_rpc_t_gates"] = gate_ct_tracker[0]
        circuit_stats["optimized_rpc_t_gates"] = gate_ct_tracker[-1]
        if gate_ct_tracker[0] > 0:
            circuit_stats["t_optimize_ratio_rpc"] = (
                gate_ct_tracker[0] - gate_ct_tracker[-1]
            ) / gate_ct_tracker[0]
        else:
            circuit_stats["t_optimize_ratio_rpc"] = 0.0
        circuit_stats["optimize_iterations_rpc"] = len(gate_ct_tracker) - 1
    else:
        circuit_stats["processed_rpc_t_gates"] = "N/A"
        circuit_stats["optimized_rpc_t_gates"] = "N/A"
        circuit_stats["t_optimize_ratio_rpc"] = "N/A"
        circuit_stats["optimize_iterations_rpc"] = 0

    if rpc_stats_list:
        raw_stats = rpc_stats_list[0]
        opt_stats = rpc_stats_list[-1]
        circuit_stats["raw_rpc_t_layers"] = raw_stats.get("t layers", "N/A")
        circuit_stats["optimized_rpc_t_layers"] = opt_stats.get("t layers", "N/A")
        raw_t_layers_count = raw_stats.get("t layers", 0)
        opt_t_layers_count = opt_stats.get("t layers", 0)

        if raw_t_layers_count > 0:
            circuit_stats["t_layer_reduction_ratio_rpc"] = 1.0 - (
                opt_t_layers_count / raw_t_layers_count
            )
        else:
            circuit_stats["t_layer_reduction_ratio_rpc"] = 0.0

    return circuit_stats


def _analyze_rpc_state(
    t_layers: List[TableauForGate], measure_tab: TableauForGate, num_qubits: int
) -> Dict[str, Any]:
    """A single, consistent function to analyze the state of an RPC object."""
    stats = {}

    stats["rotation_operators"] = sum(layer.stab_counts for layer in t_layers if layer)
    stats["rotation_layers"] = len(
        [layer for layer in t_layers if layer and layer.stab_counts > 0]
    )
    stats["measurement_operators"] = measure_tab.stab_counts if measure_tab else 0
    stats["avg_rotations_per_layer"] = (
        (stats["rotation_operators"] / stats["rotation_layers"])
        if stats["rotation_layers"] > 0
        else 0.0
    )

    all_pauli_weights = []
    qubit_interaction_degree = np.zeros(num_qubits, dtype=int)

    for op_list in [t_layers, [measure_tab]]:
        for op in op_list:
            if op is None or op.stab_counts == 0:
                continue
            x_part = op.tableau[:, :num_qubits]
            z_part = op.tableau[:, num_qubits:-1]
            nontrivial_paulis = x_part | z_part
            individual_weights = np.sum(nontrivial_paulis, axis=1)
            all_pauli_weights.extend(individual_weights.tolist())

            for i in range(nontrivial_paulis.shape[0]):
                if individual_weights[i] > 1:
                    active_qubits = np.where(nontrivial_paulis[i, :])[0]
                    qubit_interaction_degree[active_qubits] += 1

    if all_pauli_weights:
        stats["avg_operator_pauli_weight"] = float(np.mean(all_pauli_weights))
        stats["std_operator_pauli_weight"] = float(np.std(all_pauli_weights))
        stats["max_operator_pauli_weight"] = int(np.max(all_pauli_weights))
    else:
        stats.update(
            {
                "avg_operator_pauli_weight": 0.0,
                "std_operator_pauli_weight": 0.0,
                "max_operator_pauli_weight": 0,
            }
        )

    if num_qubits > 0 and np.any(qubit_interaction_degree):
        stats["avg_qubit_interaction_degree"] = float(np.mean(qubit_interaction_degree))
        stats["std_qubit_interaction_degree"] = float(np.std(qubit_interaction_degree))
        stats["max_qubit_interaction_degree"] = int(np.max(qubit_interaction_degree))
    else:
        stats.update(
            {
                "avg_qubit_interaction_degree": 0.0,
                "std_qubit_interaction_degree": 0.0,
                "max_qubit_interaction_degree": 0,
            }
        )

    return stats


# --- Main Conversion Function ---


def convert_to_pbc_circuit(
    clifford_t_circuit: QuantumCircuit,
    optimize_pbc: bool = False,
    optimize_t_maxiter: int = 5,
    if_print_rpc: bool = False,
    layering_method: str = "v2",
    layering_max_checks: Optional[int] = None,
    output_prefix: Optional[str] = None,
    max_workers: Optional[int] = None,
    use_nwqec: bool = True,
) -> tuple[QuantumCircuit, dict]:
    """Unified entry point for PBC conversion (nwqec when available, else parallel RPC)."""
    start_time = time.time()
    n_qubits = clifford_t_circuit.num_qubits
    qreg = QuantumRegister(n_qubits, "q")
    pbc_qc = QuantumCircuit(qreg)
    ops_counts = clifford_t_circuit.count_ops()
    initial_clifford_t_t_count = ops_counts.get("t", 0) + ops_counts.get("tdg", 0)

    if if_print_rpc:
        print("[PBC] Starting PBC conversion")

    # Prefer nwqec PBC when available (bypasses Python RPC path)
    if use_nwqec and is_nwqec_available():
        pbc_qc, basic_stats = transpile_to_pbc_cpp(
            clifford_t_circuit,
            is_file=False,
            t_opt=optimize_pbc and optimize_t_maxiter > 0,
            keep_cx=False,
            forbid_python_fallback=False,
        )
        fuse_applied = basic_stats.get("pbc_fuse_t_applied", False)
        pbc_stats = {
            **basic_stats,
            "num_qubits": clifford_t_circuit.num_qubits,
            "initial_clifford_t_t_gates_for_pbc": initial_clifford_t_t_count,
            "pbc_conversion_time": time.time() - start_time,
            "pbc_optimized": bool(
                optimize_pbc and optimize_t_maxiter > 0 and fuse_applied
            ),
        }
        return pbc_qc, pbc_stats

    # Always use parallel RPC for Python fallback (unless singleton disables optimization)
    use_parallel_rpc = layering_method != "singleton"
    if if_print_rpc:
        print(
            "[PBC] Using ParallelRotationPauliCirc"
            if use_parallel_rpc
            else "[PBC] Using RotationPauliCirc (Singleton/No opt)"
        )
    rpc = (
        ParallelRotationPauliCirc(clifford_t_circuit, max_workers=max_workers)
        if use_parallel_rpc
        else RotationPauliCirc(clifford_t_circuit)
    )

    if rpc.process(ifprint=if_print_rpc):
        raise RuntimeError(
            "Error during RPC processing. Circuit may have no T-gates or unsupported gates."
        )

    # If singleton layering is selected, skip any optimization later
    effective_layering_method = layering_method
    if layering_method == "singleton":
        if if_print_rpc:
            print("[PBC] Singleton layering selected; optimization will be skipped.")

    rpc.layering(
        method=effective_layering_method,
        ifprint=if_print_rpc,
        max_layer_checks=layering_max_checks,
    )
    pre_opt_stats = _analyze_rpc_state(rpc.t_layers, rpc.measure_tab, rpc.num_qubits)
    pre_opt_summary_stats = {f"pre_opt_{k}": v for k, v in pre_opt_stats.items()}
    pre_opt_summary_stats["pre_opt_t_gates"] = rpc.t_tab.stab_counts

    if output_prefix:
        output_dir = os.path.dirname(output_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        save_pbc_layers_txt(rpc.t_layers, f"{output_prefix}_pre_opt_tlayers.txt")
        save_pbc_measurement_basis_txt(
            rpc.measure_tab, f"{output_prefix}_pre_opt_measure_basis.txt"
        )

    # Skip optimization entirely for singleton layering or when optimization is disabled
    if layering_method == "singleton" or (not optimize_pbc) or optimize_t_maxiter == 0:
        if if_print_rpc:
            print("[PBC] Skipping T-merging/optimization.")
        gate_ct_tracker = [rpc.t_tab.stab_counts]
        rpc_stats_list = [rpc.statistics(ifprint=False)]
    else:
        gate_ct_tracker, rpc_stats_list = rpc.optimize_t(
            maxiter=optimize_t_maxiter,
            ifprint=if_print_rpc,
            layering_method=layering_method,
            stat_out=True,
        )

    # Save post-opt files (recycle pre-opt when singleton or opt disabled)
    if output_prefix:
        post_tlayers = f"{output_prefix}_post_opt_tlayers.txt"
        post_measure = f"{output_prefix}_post_opt_measure_basis.txt"
        pre_tlayers = f"{output_prefix}_pre_opt_tlayers.txt"
        pre_measure = f"{output_prefix}_pre_opt_measure_basis.txt"

        if layering_method == "singleton" or optimize_t_maxiter == 0:
            # Reuse pre-opt artifacts to avoid any redundant rebuilding
            try:
                if os.path.exists(pre_tlayers):
                    shutil.copyfile(pre_tlayers, post_tlayers)
                if os.path.exists(pre_measure):
                    shutil.copyfile(pre_measure, post_measure)
            except Exception as e:
                if if_print_rpc:
                    print(f"[PBC] Warning: failed to recycle pre-opt files: {e}")
        else:
            save_pbc_layers_txt(rpc.t_layers, post_tlayers)
            save_pbc_measurement_basis_txt(rpc.measure_tab, post_measure)

    # 3. Get Post-Opt Stats (using the unified analysis function)
    post_opt_stats = _analyze_rpc_state(rpc.t_layers, rpc.measure_tab, rpc.num_qubits)

    # --- START OF FIX ---
    # Use a simple, consistent prefix for all post-optimization keys.
    post_opt_summary_stats = {f"pbc_{k}": v for k, v in post_opt_stats.items()}
    # --- END OF FIX ---

    # 4. Aggregate ALL stats into one dictionary
    final_summary_stats = _format_pbc_optimization_results(
        rpc, initial_clifford_t_t_count, gate_ct_tracker, rpc_stats_list
    )
    final_summary_stats.update(pre_opt_summary_stats)
    final_summary_stats.update(
        post_opt_summary_stats
    )  # Now contains keys like pbc_rotation_operators
    final_summary_stats["pbc_conversion_time"] = time.time() - start_time

    # Build Qiskit circuit
    for layer_idx, t_layer_tab in enumerate(rpc.t_layers):
        if not (t_layer_tab and t_layer_tab.stab_counts > 0):
            continue
        if pbc_qc.data:
            pbc_qc.barrier(qreg)
        for i in range(t_layer_tab.stab_counts):
            pauli_row = t_layer_tab.tableau[i]
            pauli_chars = ["I"] * n_qubits
            for k in range(n_qubits):
                if pauli_row[k] and pauli_row[n_qubits + k]:
                    pauli_chars[k] = "Y"
                elif pauli_row[k]:
                    pauli_chars[k] = "X"
                elif pauli_row[n_qubits + k]:
                    pauli_chars[k] = "Z"

            active_pauli_str = "".join(p for p in pauli_chars if p != "I")
            if not active_pauli_str:
                continue
            q_indices = [k for k, p in enumerate(pauli_chars) if p != "I"]
            qargs = [qreg[k] for k in q_indices]
            angle = Rotation.PI_m8.value if pauli_row[-1] else Rotation.PI_8.value
            pbc_qc.append(PBM.generate_gate(active_pauli_str, angle), qargs)

    if rpc.measure_tab and rpc.measure_tab.stab_counts > 0:
        if pbc_qc.data:
            pbc_qc.barrier(qreg)
        for i in range(rpc.measure_tab.stab_counts):
            pauli_row = rpc.measure_tab.tableau[i]
            pauli_chars = ["I"] * n_qubits
            for k in range(n_qubits):
                if pauli_row[k] and pauli_row[n_qubits + k]:
                    pauli_chars[k] = "Y"
                elif pauli_row[k]:
                    pauli_chars[k] = "X"
                elif pauli_row[n_qubits + k]:
                    pauli_chars[k] = "Z"

            active_pauli_str = "".join(p for p in pauli_chars if p != "I")
            if not active_pauli_str:
                continue
            q_indices = [k for k, p in enumerate(pauli_chars) if p != "I"]
            qargs = [qreg[k] for k in q_indices]
            pbc_qc.append(PBM.generate_measure(active_pauli_str), qargs)

    if if_print_rpc:
        print("[PBC] PBC conversion complete.")
    return pbc_qc, final_summary_stats
