from typing import Any, Dict

import numpy as np
from qiskit import QuantumCircuit
from tqdm import tqdm

from .tab_gate import TableauForGate, TableauPauliBasis


def process_string(s):
    """
    Processes a Pauli string representation into a structured format.

    Args:
        s (str): Pauli string starting with '+' or '-' followed by Pauli operators

    Returns:
        tuple: (non_I_chars, position_list, sign)
            - non_I_chars: String of non-identity Pauli operators
            - position_list: List of positions where non-identity operators occur
            - sign: 1 for '+', -1 for '-'
    """
    # Determine the sign based on the first character
    sign = 1 if s[0] == "+" else -1

    # Extract non-'I' characters and their positions
    non_I_chars = "".join([c for i, c in enumerate(s[1:]) if c != "I"])
    position_list = [
        i for i, c in enumerate(s[1:]) if c != "I"
    ]  # Add 1 to account for the sign bit

    # Create the final tuple
    result = (non_I_chars, position_list, sign)

    return result


class RotationPauliCirc:
    """
    A class that represents a quantum circuit in terms of Pauli rotations.
    This class handles the conversion of Clifford+T circuits to Pauli-based computation (PBC) form,
    where T gates are represented as Pauli rotations and can be optimized through layering and merging.

    Key attributes:
        qc (QuantumCircuit): The input quantum circuit
        num_qubits (int): Number of qubits in the circuit
        t_tab (TableauPauliBasis): Tableau representing T-gates as Pauli rotations
        t_counts (int): Count of T-gates in the circuit
        measure_tab (TableauForGate): Tableau representing the measurement basis
        t_layers (list): List of TableauForGate objects representing layered T-gates
    """

    def __init__(self, qc: QuantumCircuit):
        """
        Initialize a RotationPauliCirc object from a quantum circuit.

        Args:
            qc (QuantumCircuit): Input quantum circuit to be converted to PBC form
        """
        self.qc = qc
        self.num_qubits = qc.num_qubits

        self.t_tab = None
        self.t_counts = None
        self.measure_tab = None
        self.t_layers = None
        return

    def process(self, ifprint=True):
        """
        Processes the input quantum circuit to convert T-gates into Pauli rotations.
        This is the first step in PBC conversion, where:
        1. T-gates are converted to Z-basis Pauli rotations
        2. Clifford gates (cx, h, s) are tracked to update the measurement basis
        3. The tableau is built in reverse order and then flipped

        Args:
            ifprint (bool): If True, prints progress for large circuits

        Returns:
            bool: True if error occurred (unsupported gates or no T-gates), False otherwise
        """
        t_ct = 0
        n_qubits = self.qc.num_qubits
        measure_mtx = np.zeros((n_qubits, 2 * n_qubits + 1), dtype=bool)
        measure_mtx[:, n_qubits : 2 * n_qubits] = np.eye(n_qubits, dtype=bool)
        self.measure_tab = TableauForGate(measure_mtx)

        t_tab = None
        clifford_basis = ("cx", "h", "s", "sdg", "x", "y", "z")
        gates = self.qc.reverse_ops()
        if ifprint:
            gates = tqdm(gates, desc="      Processing gates")

        for gate in gates:
            if gate.name == "t" or gate.name == "tdg":
                q_index = gate.qubits[0]._index
                temp_mtx = np.zeros((1, 2 * self.num_qubits + 1), dtype=bool)
                temp_mtx[0, self.num_qubits + q_index] = True  # set Z_i
                temp_mtx[0, -1] = gate.name == "tdg"
                if t_ct == 0:
                    t_tab = TableauPauliBasis(temp_mtx)
                else:
                    t_tab.append(temp_mtx)
                t_ct += 1
            elif gate.name == "measure":
                continue
            elif gate.name in ("barrier", "reset"):
                continue
            elif gate.name in clifford_basis:
                q_indices = [q._index for q in gate.qubits]
                self.measure_tab.apply_gate(gate.name, q_indices)
                if t_ct != 0:
                    t_tab.apply_gate(gate.name, q_indices)
            else:
                print("unsupported gate detected: ", gate.name)
                return True

        if t_tab is None:
            if ifprint:
                print("[PBC] No T-gates detected; skipping T-layer construction.")
            # Create an empty tableau so downstream steps can proceed gracefully.
            empty_tab = np.zeros((0, 2 * self.num_qubits + 1), dtype=bool)
            self.t_tab = TableauPauliBasis(empty_tab)
            self.t_counts = 0
            self.t_layers = []
            return False
        self.t_tab = t_tab
        ## now the tab saves Pauli pi/4 rotation in a reverse order, we would like to reverse the order
        self.t_tab.tableau = np.flip(self.t_tab.tableau, axis=0)

        # update t_counts
        self.t_counts = t_ct
        return False

    def layering(self, method="bare", ifprint=False, max_layer_checks=None):
        """
        Organizes T-gates into layers based on commutation rules.

        Methods:
        - 'bare': Basic single-pass layering that groups commuting T-gates.
        - 'v2':   Backward-scan layering that may reduce circuit depth. Accepts
                  `max_layer_checks` to bound the scan window.
        - 'singleton': One T-gate per layer (disables grouping).

        Args:
            method (str): Layering method ('bare', 'v2', or 'singleton').
            ifprint (bool): If True, shows progress information.
            max_layer_checks (int | None): Bound for 'v2' backward scan; ignored
                otherwise.
        """
        if ifprint:
            print(f"[PBC] Entering layering (method={method})...")
        if self.t_tab is None:
            self.process()
        if self.t_tab is None or self.t_tab.stab_counts == 0:
            self.t_layers = []  # Return empty list if no T gates
            return
        if method == "bare":
            self.t_layers = self.t_tab.layer()
        elif method == "v2":
            self.t_layers = self.t_tab.layer_v2(max_layer_checks=max_layer_checks)
        elif method == "singleton":
            # Each rotation becomes its own layer (no grouping)
            self.t_layers = []
            if self.t_tab is not None and self.t_tab.stab_counts > 0:
                for j in range(self.t_tab.stab_counts):
                    single_row = self.t_tab.tableau[j].reshape(
                        (1, 2 * self.num_qubits + 1)
                    )
                    self.t_layers.append(TableauPauliBasis(single_row))
        if ifprint:
            print(f"[PBC] Layering complete (method={method}).")
        return

    def t_merging(self, debug=False):
        """
        Optimizes the circuit by merging T-gates across layers.
        The process:
        1. Processes layers from last to first
        2. Simplifies each layer into Z-gates and S-gates
        3. Commutes these gates through subsequent layers
        4. Absorbs gates into the measurement basis where possible

        Args:
            debug (bool): If True, returns debugging information about Pauli gates

        Returns:
            tuple: (pauli_gates, sq_pauli_gates) if debug=True, None otherwise
        """
        if debug:
            print("[PBC] Entering T-merging (debug mode)...")
        if self.t_layers is None:
            self.layering()

        ## merging T gates in different t_layers and move them backwards
        # (1) change the later T layers, and (2) change measurement basis

        if debug:
            pauli_gates = []
            sq_pauli_gates = []

        # Add progress bar for T-gate merging
        for j in tqdm(
            range(len(self.t_layers) - 1, -1, -1),
            desc="      Merging T-gates",
            leave=False,
        ):
            tab = self.t_layers[j]
            z_gates, s_gates = tab.simplify()

            # commute Pauli and Puali-pi/2 rotation gates to the end of the circuit
            for k in range(j + 1, len(self.t_layers)):
                tab_follows = self.t_layers[k]
                for z in z_gates:
                    # commute Pauli gates through
                    tab_follows.commute_pauli(z)
                for s in s_gates:
                    tab_follows.front_multiply_pauli(s)
                self.t_layers[k] = tab_follows

            ## absorb these gates by the measurements
            for z in z_gates:
                self.measure_tab.commute_pauli(z)
            for s in s_gates:
                self.measure_tab.front_multiply_pauli(s)

            if debug:
                pauli_gates.append(z_gates)
                sq_pauli_gates.append(s_gates)

        if debug:
            print("[PBC] T-merging complete (debug mode).")
            return pauli_gates, sq_pauli_gates

    @staticmethod
    def check_identical_paulis(tab1: TableauForGate, tab2: TableauForGate):
        """
        Checks for identical Pauli strings between two tableaus.
        Used to identify opportunities for gate merging.

        Args:
            tab1 (TableauForGate): First tableau to compare
            tab2 (TableauForGate): Second tableau to compare

        Returns:
            tuple: (total_count, tracking_dict)
                - total_count: Number of identical Pauli strings found
                - tracking_dict: Maps indices from tab1 to matching indices in tab2
        """
        total_ct = 0
        tracking: Dict[int, Any] = {}
        if tab1.stab_counts == 0 or tab2.stab_counts == 0:
            return total_ct, tracking

        for j in range(tab1.stab_counts):
            line_1 = tab1[j][:-1]
            indices = []
            for k in range(tab2.stab_counts):
                line_2 = tab2[k][:-1]
                if not np.any(line_1 ^ line_2):
                    ## these two lines have the same Pauli strings
                    indices.append(k)
                    total_ct += 1
            tracking[j] = indices

        return total_ct, tracking

    def t_merging_after(self, debug=True):
        """
        Placeholder for post-merging optimization.
        Currently not implemented as the main t_merging process
        handles all necessary optimizations.
        """
        return

    def update_tableau(self):
        """
        Reconstructs the main tableau (t_tab) from the layered representation.
        This is called after optimization to update the circuit representation.
        If no layers exist, creates an empty tableau.
        """
        if self.t_layers is None or len(self.t_layers) == 0:
            # If no layers, create an empty tableau with the right shape
            self.t_tab = TableauPauliBasis(
                np.zeros((0, 2 * self.num_qubits + 1), dtype=bool)
            )
            return

        temp_mtx = self.t_layers[0].tableau.copy()
        for tab in self.t_layers[1:]:
            temp_mtx = np.append(temp_mtx, tab.tableau, axis=0)

        self.t_tab = TableauPauliBasis(temp_mtx)
        return

    def optimize_t(
        self, maxiter=10, ifprint=False, stat_out=False, layering_method="v2"
    ):
        """
        Main optimization routine that iteratively applies layering and merging
        to reduce the number of T-gates in the circuit.

        The process:
        1. Applies layering to organize T-gates
        2. Performs T-gate merging
        3. Updates the tableau
        4. Repeats until no improvement or max iterations reached

        Args:
            maxiter (int): Maximum number of optimization iterations
            ifprint (bool): If True, prints progress information
            stat_out (bool): If True, returns detailed statistics
            layering_method (str): Method to use for layering ('bare' or 'v2')

        Returns:
            If stat_out=True:
                tuple: (gate_ct_tracker, stats_list)
                    - gate_ct_tracker: List of T-gate counts after each iteration
                    - stats_list: List of statistics dictionaries for each iteration
            Otherwise:
                list: gate_ct_tracker only
        """
        if ifprint:
            print("[PBC] Starting optimize_t loop...")
        iter = 0
        gate_ct_tracker = [self.t_tab.stab_counts]
        stats = []

        # If no T gates, return early with empty stats
        if self.t_tab is None or self.t_tab.stab_counts == 0:
            if stat_out:
                return gate_ct_tracker, [self.statistics(ifprint=ifprint)]
            return gate_ct_tracker

        improve_ct = 1

        if stat_out:
            stats.append(self.statistics(ifprint=ifprint))

        if ifprint:
            _ = tqdm(total=maxiter, desc="PBC Optimization", leave=True)
        while iter < maxiter:
            if ifprint:
                print(f"[PBC] optimize_t Iteration {iter+1}/{maxiter}: Layering...")
            self.t_layers = []
            self.layering(method=layering_method, ifprint=ifprint)
            if ifprint:
                print(f"[PBC] optimize_t Iteration {iter+1}/{maxiter}: Merging...")
            self.t_merging()
            ct = 0
            for tab in self.t_layers:
                ct += len(tab.tableau)
            improve_ct = gate_ct_tracker[-1] - ct
            self.update_tableau()
            if ifprint:
                print(
                    f"[PBC] optimize_t Iteration {iter+1}/{maxiter}: Tableau updated. T-count: {ct}"
                )
            if improve_ct == 0:
                if ifprint:
                    print(
                        f"[PBC] No improvement, stopping optimize_t at iteration {iter+1}."
                    )
                break
            if ct == 0:
                if ifprint:
                    print(
                        f"[PBC] No T-gates remaining, stopping optimize_t at iteration {iter+1}."
                    )
                break
            gate_ct_tracker.append(ct)

            if stat_out:
                # Append stats for the *newly optimized* state
                stats.append(self.statistics())

            iter += 1
            if ifprint:
                print(f"[PBC] optimize_t Iteration {iter+1}/{maxiter} complete.")
        if ifprint:
            print("[PBC] optimize_t loop complete.")
        if stat_out:
            return gate_ct_tracker, stats
        else:
            return gate_ct_tracker

    def statistics(self, ifprint=False):
        """
        Generates detailed statistics about the current state of the circuit.
        Collects information about:
        - Number of T-layers
        - Gate counts per layer
        - Pauli weights and their distribution
        - Qubit occupation statistics

        Args:
            ifprint (bool): If True, shows progress information during layering

        Returns:
            dict: Statistics including:
                - t layers: Number of T-layers
                - gate counts: Total number of gates
                - layer gate count: List of gates per layer
                - layer weights: Average Pauli weights per layer
                - layer occupation: Qubit occupation per layer
                - pauli weights: Average Pauli weight across all gates
                - total occupation: Average qubit occupation
        """
        if ifprint:
            print("[PBC] Collecting statistics...")

        if self.t_layers is None:
            self.layering("v2", ifprint=ifprint)

        stat = {}
        stat["t layers"] = len(self.t_layers)

        total_occupation = np.zeros(self.num_qubits, dtype=int)
        layer_cts = []

        # --- START OF FIX ---
        # We will now store ALL individual weights, not layer averages.
        all_individual_weights = []
        # --- END OF FIX ---

        # For backward compatibility, we can still calculate layer averages if needed, but not use them for overall stats.
        layer_avg_weights = []
        layer_occupation = []

        ct = 0
        for tab in self.t_layers:
            this_layer_ct = len(tab.tableau)
            if this_layer_ct == 0:
                continue

            layer_cts.append(this_layer_ct)

            xmtx = tab.tableau[:, : tab.qubits]
            zmtx = tab.tableau[:, tab.qubits : -1]
            nontrivial_paulis = xmtx | zmtx

            # Calculate individual weights for this layer
            individual_weights_in_layer = np.sum(nontrivial_paulis, axis=1)

            # Append the individual weights to the master list
            all_individual_weights.extend(individual_weights_in_layer.tolist())

            # For backward compatibility / detailed layer-by-layer stats if needed
            layer_avg_weights.append(np.mean(individual_weights_in_layer))

            occupation = np.sum(nontrivial_paulis, axis=0)
            layer_occupation.append(occupation / this_layer_ct)
            total_occupation += occupation
            ct += this_layer_ct

        stat["gate counts"] = ct
        stat["layer gate count"] = layer_cts
        stat["layer occupation"] = layer_occupation

        # --- CALCULATE STATS FROM THE CORRECT (FLAT) LIST ---
        if all_individual_weights:
            stat["pauli weights"] = np.mean(
                all_individual_weights
            )  # This is the true average weight
            stat["layer weights"] = (
                all_individual_weights  # Pass the full list for std/max/min calculation later
            )
        else:
            stat["pauli weights"] = 0.0
            stat["layer weights"] = []  # Pass an empty list

        if ct != 0:
            stat["total occupation"] = total_occupation / ct
        else:
            stat["total occupation"] = np.zeros((self.num_qubits,), dtype=int)

        return stat
