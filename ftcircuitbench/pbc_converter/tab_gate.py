from typing import Optional

import numpy as np
from tqdm import tqdm

# OPERATOR_CONST = ['I','Z','X','Y']
OPERATOR_CONST = {
    0: "I",
    1: "Z",
    2: "X",
    3: "Y",
    "I": 0,
    "Z": 1,
    "X": 2,
    "Y": 3,
    "i": 0,
    "z": 1,
    "x": 2,
    "y": 3,
}
ANTI_SYM_TENSOR = {(1, 2): 0, (2, 3): 0, (3, 1): 0, (2, 1): 1, (3, 2): 1, (1, 3): 1}
SINGLE_GATE_SET = set({"X", "x", "Y", "y", "Z", "z", "H", "h", "S", "s"})
TWO_GATE_SET = set({"cnot", "CNOT", "CZ", "cz", "SWAP", "swap"})


class TableauForGate:
    """
    This is only for tracking clifford gates applying to Pauli strings,
    There is no commutation checks to make sure the Pauli commutes with each other.
    There is no functionality support for tableau matrix reduction
    There is no support (at the moment) for qubit index relabeling and row permutations
    Only supprot the most basic operations which are necessary for gate operations.
    We just use normal numpy ndarrays (with bool type) for data storage at the moment, no sparse matrix support yet.
    """

    def __init__(self, tab):
        ## dimensional check:
        ##   1). the last bit should be to track the phase of the corresponding stabilizer operator, so column number must be odd
        ##   2). the row number should <= column number // 2
        ##
        if isinstance(tab, TableauForGate):
            self.tableau = tab.tableau
            self.qubits = tab.qubits
            self.stab_counts = tab.stab_counts
            self.shape = tab.shape
            return

        else:
            self.tableau = np.array(tab)
            row_number, col_number = self.tableau.shape

            if col_number % 2 == 0:
                raise ValueError(
                    "The input tableau dimension is incorrect. Each row should be odd."
                )
            # if row_number > col_number //2:
            #    raise ValueError('The input tableau has more rows than the qubit number.')

            self.qubits = col_number // 2
            self.stab_counts = row_number

            self.shape = self.tableau.shape

            return

    def tab(self, dtype="bool"):
        """
        This function gives the tableau matrix with the phase bits
        Return:
            sparse matrix for the tableau (with the phase bit)
            Dimension [s x (2n + 1)], where s -> stabilizer counts, n -> qubit number
        """
        if dtype == "bool":
            return self.tableau
        else:
            return self.tableau.astype(int)

    def stabilizers(self, dtype="bool"):
        """
        This function returns the tableau for the stabilizers, without the phase
        Return:
            Sparse matrix for the tableau (without phase bit)
            The sparse matrix is in coo_matrix format.
            Dimension [s x (2 n)], where s -> stabilizer counts, n -> qubit number
        """
        if dtype == "bool":
            return self.tableau[:, :-1]
        else:
            return self.tableau[:, :-1].astype(int)

    def phase(self, dtype="bool"):
        """
        This function returns the phase bit for each stabilizers
        Return:
            Sparse matrix in coo_matrix format
            Dimension: [s x 1], where s -> stabilizer counts
        """
        if dtype == "bool":
            return self.tableau[:, -1]
        else:
            return self.tableau[:, -1].astype(int)

    def get_index(self, row=None, input_stab=None):
        """
        This function calculate the index matrix for the stabilizer,
        we assign each single-qubit Pauli operator an index, which is given in the order of OPERATOR_CONST
        Input:
            row: the row number in the current instance that we want to convert to index representation
            input_stab: the input stabilizer that are going to be converted

            if no input is given, the function calcualte the index representation of the tableau of the current instance
            if both row and input_stab are given, only row input will be processed.
        """
        if row is None and input_stab is None:
            stab = self.stabilizers()
            x_part = stab[:, : self.qubits]
            z_part = stab[:, self.qubits :]

            index_mtx = x_part * 2 + z_part
            return index_mtx
        elif input_stab is None:
            try:
                row = int(row)
            except Exception:
                raise TypeError("get_index row number needs to be integer.")

            stab = (
                self.tableau[row, : self.qubits] * 2
                + self.tableau[row, self.qubits : -1]
            )
            return stab
        else:
            in_stab = np.array(input_stab)
            if in_stab.shape[1] % 2 == 1:
                print("Input stab with phase bit")
                in_stab = in_stab[:, :-1]
            else:
                print("Input stab without phase bit")

            bits = in_stab.shape[1] // 2

            x_part = in_stab[:, :bits]
            z_part = in_stab[:, bits:]

            index_mtx = x_part * 2 + z_part
            return index_mtx

    def readout(self, row):
        """
        This function gives a readout output (+/- Pauli operators for each bits) of a certain row of the stabilier
        Input:
            row: the integer number for the row that want to read out in the tableau
        Output:
            str_out: a string that contains the sign and Pauli operators for each bits
        """
        try:
            row = int(row)
        except Exception:
            raise TypeError("readout need an integer row number.")
        row_tab_index = (
            self.tableau[row, : self.qubits] * 2 + self.tableau[row, self.qubits : -1]
        )
        row_tab_index = row_tab_index.flatten()

        row_tab_index_str = [OPERATOR_CONST[x] for x in row_tab_index]

        row_sign = self.tableau[row, -1]
        if row_sign == 0:
            str_out = "+"
        else:
            str_out = "-"

        str_out += "".join(row_tab_index_str)
        return str_out

    def __str__(self):
        """
        Print the qubit number.
        Print the Pauli operators for each bits
        """
        row = self.tableau.shape[0]
        str_out = "-" * max(self.qubits, 30) + "\n"
        str_out += "Qubit number: " + str(self.qubits) + "\n"
        str_out += "Stabilizer count: " + str(row) + "\n"
        str_out += "Stabilizers: \n"

        for i in range(0, row):
            str_out += self.readout(i)
            str_out += "\n"
        str_out += "-" * max(self.qubits, 30) + "\n"

        return str_out

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, i, j=None):
        if j is None:
            return self.tableau[i]
        else:
            return self.tableau[i, j]

    @staticmethod
    def convert_back(string_in):
        """
        This function convert a string of Pauli Gates, to stabilizer formalism
        Input:
            string_in:  The string of Pauli Gates to convert to the tableau formalism
        Output:
            out_stab:   The stabilizer for output
        Note:
            (1) The input string can be in upper or lower letters
            (2) The string can be start from + or -, which will give the phase bit
            (3) If the string does not start by the sign, we will assume the sign is +
            (4) The input can also be a list of string. The output will be a list of Tableau instances for each line
            (5) If the input is neither a string, nor a list of string (list or ndarray), a TypeError will be raised.
        """
        if isinstance(string_in, str):
            ## consider the first char, if it is a sign character
            phase_bit = 0
            gates = string_in
            if string_in[0] == "+" or string_in[0] == "-":
                gates = string_in[1:]
                if string_in[0] == "-":
                    phase_bit = 1
            try:
                index_list = [OPERATOR_CONST[x] for x in gates]
            except KeyError:
                raise KeyError("Input string needs to be only { X, Y, Z, I }.")

            index_list = np.array(index_list)
            tab = np.zeros(len(index_list) * 2 + 1)
            tab[-1] = phase_bit
            tab[: len(index_list)] = index_list // 2
            tab[len(index_list) : -1] = index_list % 2
            tab = tab.astype(bool)
            if len(tab.shape) == 1:
                tab = tab.reshape((1, tab.shape[0]))
            # print(tab)

            out_tab = TableauForGate(tab)
            return out_tab
        elif isinstance(string_in, list) or isinstance(string_in, np.ndarray):
            ## Here we assume the input
            out_tab_list = []
            for string in string_in:
                out_tab_list.append(TableauForGate.convert_back(string))

            return out_tab_list
        else:
            raise TypeError("The input type can only be string, list, ndarray.")

    @staticmethod
    def binary_add(mtx1, mtx2):
        """
        This function provide a general method for adding two sparse matrix togather in mod 2
        """
        if not isinstance(mtx1, np.ndarray):
            mtx1 = np.array(mtx1).astype(bool)
        elif mtx1.dtype != np.bool_:
            mtx1 = mtx1.astype(bool)

        if not isinstance(mtx2, np.ndarray):
            mtx2 = np.array(mtx2).astype(bool)
        elif mtx2.dtype != np.bool_:
            mtx2 = mtx2.astype(bool)

        mtx3 = mtx1 ^ mtx2

        return mtx3

    # @staticmethod
    # def commute(line1, line2, commutation_out=False):
    #     """
    #     This function tests the commutation relation of two stabilizers given.
    #     Here we assume that the input are the tableau representation (with phase)
    #     In the function, we test the size of the two lines, if they did not match, a ValueError will be raised.
    #     Input:
    #         line1, line2: input tableau of two stabilizers with phase bits
    #     Output:
    #         Boolean type
    #     """

    #     line1 = np.array(line1)
    #     line2 = np.array(line2)

    #     ## test length of the two tableau
    #     if line1.shape[1] != line2.shape[1]:
    #         raise ValueError("The input stabilizers have different qubit size.")

    #     stab1 = line1[0, :-1]
    #     stab2 = line2[0, :-1]

    #     bits = (stab1.shape[1] - 1) // 2

    #     # permute X and Z part of stab2
    #     stab2_new = np.zeros_like(stab2, dtype=int)
    #     stab2_new[:, :bits] = stab2[:, bits:]
    #     stab2_new[:, bits] = stab2[:, :bits]

    #     res = (
    #         stab1 @ stab2_new.T
    #     ) & 1  # using bitwise operation for element-wise mod 2
    #     if not commutation_out:
    #         if res == 0:
    #             return True
    #         else:
    #             return False
    #     else:
    #         return res

    def is_commute(self, stab_in, commutation_out=False):
        """
        This function calculate whether the given stabilizer(s) commute with the current tableau
        Will give a list of binary values,  0 -> commute
                                            1 -> anti-commute
        The first dimension corresponds to the input stabilizer(s)
        The second dimension corresponds to the self stabilizers

        Input:
            stab_in: this is the stabilizer(s) input, which is to determine the commutation relation with the current tableau
                If the stab_in is a Tableau class instance, we directly calculate the commutation relation
                If the stab_in is not a Tableau instance, it will be converted into Tableau instance at first.
        Output:
            commute_relation: the resulting commutation relation, 0 -> commute, 1 -> anti-commute
        Note:
            If the two tableaus have different qubit numbers, the smaller tableau will be extended by appending extra I's.
            A warning will be printed to terminal.
        """

        if isinstance(stab_in, TableauForGate):
            stab_mtx = stab_in.stabilizers()
            n_qubit = stab_in.qubits
        elif isinstance(stab_in, np.ndarray):
            if len(stab_in.shape) == 1:
                stab_in = stab_in.reshape((1, stab_in.shape[0]))

            n_qubit = len(stab_in[0]) >> 1
            stab_mtx = stab_in[:, : 2 * n_qubit]

        else:
            stab_in = np.array(stab_mtx)
            n_qubit = len(stab_mtx[0]) >> 1
            stab_mtx = stab_in[:, 2 * n_qubit]

        if n_qubit != self.qubits:
            raise ValueError("The two stablizers do not have the same qubit numbers.")

        stab_mtx_temp = np.zeros_like(stab_mtx, dtype=int)
        stab_mtx_temp[:, :n_qubit] = stab_mtx[:, n_qubit:]  # Z part
        stab_mtx_temp[:, n_qubit:] = stab_mtx[:, :n_qubit]  # X part

        stab = self.stabilizers()
        commutation = (stab_mtx_temp @ stab.T) & 1

        if commutation_out:
            return np.sum(commutation) == 0, commutation
        else:
            return np.sum(commutation) == 0

    def _swap_xz(self, row_or_rows):
        """Return the X<->Z-swapped stabilizer block of a row (or rows).

        Used by `layer_v2` to precompute the swap once per new Pauli and reuse it
        across the backward layer scan, instead of repeating the swap inside
        every `is_commute` call.

        Input shape: (..., 2*n+1) bool. Output shape: (rows, 2*n) int.
        """
        arr = np.asarray(row_or_rows)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n = self.qubits
        out = np.empty((arr.shape[0], 2 * n), dtype=int)
        out[:, :n] = arr[:, n : 2 * n]  # Z part
        out[:, n:] = arr[:, :n]  # X part
        return out

    def _commutes_with_swapped(self, swapped_2nq):
        """Fast commutation check for an already-XZ-swapped row block.

        `swapped_2nq` must have shape (rows, 2*n) and dtype int. Returns True iff
        every input row commutes with every stabilizer in this tableau.
        """
        commutation = (swapped_2nq @ self.stabilizers().T) & 1
        return not np.any(commutation)

    def append(self, stab_in):
        """
        This function append a new stabilizer into the tableau.
        WE DID NOT CHECK THE COMMUTATION
        The new stabilizers are required to have the same qubit number as the tableau qubit number
        Input:
            stab_in:    The stabilizer(s) that will be appended into the Tableau
        Output:
            None:       self.tableau will be changed
        Exception:
            ValueError: if the input stabilizer does not have the same qubit number
                        if the input stab_in cannot be converted into Tableau class instance
        """
        ## check the new line dimensions:
        if not isinstance(stab_in, TableauForGate):
            if not isinstance(stab_in, np.ndarray):
                stab_in = np.array(stab_in).astype(bool)

            if stab_in.dtype != np.bool_:
                stab_in = stab_in.astype(bool)

            new_mtx = np.append(self.tableau, stab_in, axis=0)

        else:
            ## Check the qubit number of the input stabilizer(s)
            if stab_in.qubits != self.qubits:
                raise ValueError(
                    "Tableau.append function requires the input stabilizers have the same qubit numbers."
                )

            new_mtx = np.append(self.tableau, stab_in.tableau, axis=0)

        self.tableau = new_mtx
        self.stab_counts = new_mtx.shape[0]

        return

    def h(self, index):
        ## dealing sign bit
        self.tableau[:, -1] = self.tableau[:, -1] ^ (
            self.tableau[:, index] * self.tableau[:, self.qubits + index]
        )
        ## dealing X and Z bits
        temp = self.tableau[:, index].copy()
        self.tableau[:, index] = self.tableau[:, self.qubits + index]
        self.tableau[:, self.qubits + index] = temp
        return

    def s(self, index):
        ## dealing with the phase bit
        self.tableau[:, -1] = self.tableau[:, -1] ^ (
            self.tableau[:, index] * self.tableau[:, self.qubits + index]
        )

        ## dealing with the z bit
        self.tableau[:, self.qubits + index] = (
            self.tableau[:, self.qubits + index] ^ self.tableau[:, index]
        )
        return

    def sdg(self, index):
        # S†: X -> -Y, Y -> X, Z -> Z. Equivalent to S applied 3 times.
        # Phase flips iff (X_i & ~Z_i) — derivable from composing S thrice.
        self.tableau[:, -1] = self.tableau[:, -1] ^ (
            self.tableau[:, index] & (~self.tableau[:, self.qubits + index])
        )
        self.tableau[:, self.qubits + index] = (
            self.tableau[:, self.qubits + index] ^ self.tableau[:, index]
        )
        return

    def x(self, index):
        # X anticommutes with Z; flips phase where Z_i = 1.
        self.tableau[:, -1] = self.tableau[:, -1] ^ self.tableau[:, self.qubits + index]
        return

    def y(self, index):
        # Y anticommutes with both X and Z; flips phase where (X_i XOR Z_i).
        self.tableau[:, -1] = self.tableau[:, -1] ^ (
            self.tableau[:, index] ^ self.tableau[:, self.qubits + index]
        )
        return

    def z(self, index):
        # Z anticommutes with X; flips phase where X_i = 1.
        self.tableau[:, -1] = self.tableau[:, -1] ^ self.tableau[:, index]
        return

    def cx(self, ctrl, targ):
        x_ctrl = self.tableau[:, ctrl]
        z_ctrl = self.tableau[:, self.qubits + ctrl]
        x_targ = self.tableau[:, targ]
        z_targ = self.tableau[:, self.qubits + targ]

        ## dealing with the phase bi
        sign_change = x_ctrl * z_targ * (x_targ ^ z_ctrl ^ 1)
        self.tableau[:, -1] = self.tableau[:, -1] ^ sign_change

        ## dealing with x and z bits
        self.tableau[:, targ] = x_ctrl ^ x_targ
        self.tableau[:, self.qubits + ctrl] = z_ctrl ^ z_targ
        return

    def apply_gate(self, gate_name, q_indices):
        if gate_name == "h":
            self.h(q_indices[0])
        elif gate_name == "s":
            self.s(q_indices[0])
        elif gate_name == "sdg":
            self.sdg(q_indices[0])
        elif gate_name == "x":
            self.x(q_indices[0])
        elif gate_name == "y":
            self.y(q_indices[0])
        elif gate_name == "z":
            self.z(q_indices[0])
        elif gate_name == "cx":
            self.cx(q_indices[0], q_indices[1])
        else:
            print(f"Gate not implemented: {gate_name}")

        return

    @staticmethod
    def g_fun(x1, z1, x2, z2):
        """
        g function for row-sum
        stab1, stab2 are two boolean np.array with sign bits
        """
        temp = (
            (x1 & z1) * (1 * z2 - x2)
            + (x1 & (z1 ^ 1)) * z2 * (2 * x2 - 1)
            + ((x1 ^ 1) & z1) * (x2 * (1 - 2 * z2))
        )
        if len(temp.shape) == 1:
            return np.sum(temp)
        else:
            return np.sum(temp, axis=1)

    def rowsum(self, stab1, stab2, tab_out=True):
        """
        compute the rowsum function of two rows
        Note:
        we make some modification of it:
        if two pauli strings commute, the sign bit will just track the sign of the multiplication of stab1 * stab2
        otherwise, the sign bit will be the sign of i*stab1*stab2
        """
        x1 = stab1[0, : self.qubits]
        z1 = stab1[0, self.qubits : -1]
        x2 = stab2[0, : self.qubits]
        z2 = stab2[0, self.qubits : -1]

        g_val = self.g_fun(x1, z1, x2, z2) & 3
        if g_val & 1:
            g_val += 1
        g_val = bool(g_val & 1)

        stab_new = stab1 ^ stab2
        stab_new[0, -1] = stab1[0, -1] ^ stab2[0, -1] ^ g_val

        if tab_out:
            return TableauForGate(stab_new)
        else:
            return stab_new

    @staticmethod
    def pauli_product(stab1, stab2):
        """
        two pauli string's product: tab1 * tab2
        bool np.arrays as input
        output bool np.arrays
        """
        nq = len(stab1) >> 1
        x1 = stab1[0, :nq]
        z1 = stab1[0, nq:-1]
        x2 = stab2[0, :nq]
        z2 = stab2[0, nq:-1]

        g_val = TableauForGate.g_fun(x1, z1, x2, z2) & 3
        if g_val & 1:
            g_val += 1
        g_val = bool(g_val & 1)

        stab_new = stab1 ^ stab2
        stab_new[0, -1] = stab1[0, -1] ^ stab2[0, -1] ^ g_val

        return stab_new

    @staticmethod
    def pauli_product_tab(tab1, tab2):
        """
        Two inputs are tableaus with only one Pauli matrix
        """
        stab1 = tab1.tableau
        stab2 = tab2.tableau

        stab_new = TableauForGate.pauli_product(stab1, stab2)
        return TableauForGate(stab_new)

    def front_multiply_pauli(self, new_pauli_tab):
        """
        compute: (i * new_pauli) * current_tableau
        and save the new tableau into the current tableau class.

        Note:
        if two pauli strings commute, the sign bit will just track the sign of the multiplication of stab1 * stab2
        otherwise, the sign bit will be the sign of i*stab1*stab2

        """
        if isinstance(new_pauli_tab, TableauForGate):
            new_pauli_mtx = new_pauli_tab.tableau[0]
        else:
            new_pauli_mtx = np.array(new_pauli_tab, dtype=bool)
            shape = new_pauli_mtx.shape
            if len(shape) > 1:
                new_pauli_mtx = new_pauli_mtx[0]

        stab2 = self.stabilizers()
        x1 = new_pauli_mtx[: self.qubits]
        z1 = new_pauli_mtx[self.qubits : -1]
        x2 = stab2[:, : self.qubits]
        z2 = stab2[:, self.qubits :]

        g_val = self.g_fun(x1, z1, x2, z2) & 3
        mask = (g_val & 1).astype(bool)  ## commute -> 0, anticommute->1
        g_val[np.where(mask == 1)] += 1

        g_val = g_val // 2

        stab1_expand = np.outer(mask, new_pauli_mtx[: 2 * self.qubits])

        stab_new = stab1_expand ^ stab2
        self.tableau[mask, -1] = (
            new_pauli_mtx[-1] ^ (self.tableau[mask, -1] ^ g_val[mask])
        ) & 1

        self.tableau[:, : 2 * self.qubits] = stab_new

    def commute_pauli(self, front_pauli):
        """
        when there is a pauli, we compute the new pauli strings that satisfies:
        front_pauli * self = new_self * front_pauli
        Note:
            front pauli only has one Pauli string
        """

        commute, commutation_relation = self.is_commute(front_pauli, True)
        if commute:
            return
        else:
            commutation_relation = commutation_relation.flatten()
            anti_row_mask = np.where(commutation_relation == 1)
            self.tableau[anti_row_mask, -1] = self.tableau[anti_row_mask, -1] ^ 1
            return


## pi/8 transformation dictionary:
TSZ_TRANSFORM = {
    0: (0, 0, 0),
    2: (0, 1, 0),
    4: (1, 0, 0),
    6: (1, 1, 0),
    1: (0, 0, 1),
    3: (1, 0, -1),
    5: (1, 0, 1),
    7: (0, 0, -1),
}


class TableauPauliBasis(TableauForGate):
    """
    Extending the tableau functionality:
    include Pauli manipulations, when there are Clifford gate operations that want to commute through the clifford operations.
    These are specific operations, only for tracking the Pauli rotations for transpilation purpose.
    """

    def __init__(self, tab):
        super().__init__(tab)
        return

    def count_unique_paulis(self):
        """
        count the number of unique pauli strings in the current tableau after the gate merging
        we totally ignore the Pauli commutation relations.
        This gives a lower bound on T-type rotation gates after gate merging.
        """
        _, ct = np.unique(self.tableau[:, :-1], axis=0, return_counts=True)
        return np.sum((ct & 1))

    def simplify(self):
        """
        Assume the current tableau is a layer of Pauli rotation gates (mutally commute)
        find the identical puali strings inside the current tableau
        If we find a pair:
            we merge them togather in the following way:
                1. if the two identical pauli strings have the same sign: output the pauli string and remove it from the tableau
                2. if the two PS have opposite sign: remove then from the tableau withoug returning anything
        If we did not find a pair:
            return nothing and do nothing to the tableau

        ## this function may need further optimization for performance.
        """

        stab_part = self.stabilizers()
        shape = (1, 2 * self.qubits + 1)

        # Find unique rows, their indices, and counts
        unique_rows, unique_indices, indices, counts = np.unique(
            stab_part,
            axis=0,
            return_inverse=True,
            return_counts=True,
            return_index=True,
        )

        # Identify the rows that are duplicated
        duplicate_mask = counts > 1
        duplicate_rows = unique_rows[duplicate_mask]

        # Map indices of duplicates
        duplicates_indices = {
            tuple(row): np.where((stab_part == row).all(axis=1))[0]
            for row in duplicate_rows
        }

        # Get unique rows excluding duplicates
        unique_only_mask = counts == 1
        ur = self.tableau[unique_indices]
        unique_only_rows = ur[unique_only_mask]  ## no repetition rows

        pauli_gates = []
        sqrt_pauli_gates = []

        for key, val in duplicates_indices.items():
            current_pauli = self.tableau[val[0]].copy()

            sign_bits = self.tableau[val, -1]
            total_rotations = len(sign_bits) - 2 * np.sum(sign_bits)
            total_rotations = total_rotations % 8  ## total number of T-type rotations

            pauli_ct, sqrt_pauli_ct, t_ct = TSZ_TRANSFORM[total_rotations]
            if t_ct != 0:
                current_pauli[-1] = t_ct < 0
                unique_only_rows = np.append(
                    unique_only_rows, current_pauli.reshape(shape), axis=0
                )  ## add the T-type gate into the tableau for tracking

            if pauli_ct != 0:
                current_pauli[-1] = 0
                # pauli_gates.append(current_pauli.reshape(shape).copy())
                pauli_gates.append(current_pauli.copy())

            if sqrt_pauli_ct != 0:
                current_pauli[-1] = 0
                # sqrt_pauli_gates.append(current_pauli.reshape(shape).copy())
                sqrt_pauli_gates.append(current_pauli.copy())

        self.tableau = unique_only_rows
        self.stab_counts = len(self.tableau)

        return pauli_gates, sqrt_pauli_gates

    def check_identical(self, new_pauli_mtx):
        """
        Check if the given np.array matches any row in the tableau (having the same Pauli string)
        ## if the sign is different, then these two
        """
        pass

    def layer(self):
        """
        Form a list of Pauli pi/4 rotations, try to layer them into commuting rotations.
        The strategy is:
        1. keep a tabuleau class for appending new Pauli rotation basis, if the new Pauli string commutes with
        """
        ## now the most basic methods.
        t_layers = []
        tab_now = self.tableau[0]
        tab_now = tab_now.reshape((1, len(tab_now)))
        tab_now = TableauPauliBasis(tab_now)

        # Add progress bar for layering
        for j in tqdm(
            range(1, self.stab_counts), desc="      Layering T-gates", leave=False
        ):
            tab_temp = self.tableau[j]
            if tab_now.is_commute(tab_temp):
                tab_now.append(tab_temp.reshape((1, len(tab_temp))))
            else:
                t_layers.append(tab_now)
                tab_now = TableauPauliBasis(tab_temp.reshape((1, len(tab_temp))))

        t_layers.append(tab_now)

        return t_layers

    def layer_v2(self, max_layer_checks: Optional[int] = None):
        """use a costly but maybe better way of grouping commuting layers

        If max_layer_checks is provided, only inspect up to that many layers
        from the end when determining insertion position. This bounds the number
        of commutation checks per insertion while preserving operation order.
        """
        n = self.qubits
        shape = (1, 2 * n + 1)

        # Pack the X|Z support of every input row into Python ints once. A
        # Python int with one bit per qubit lets the per-layer prune below run
        # as a single int-AND + truthiness check, much cheaper than numpy ops.
        x_part = self.tableau[:, :n]
        z_part = self.tableau[:, n : 2 * n]
        support_bool = x_part | z_part  # (N, n)
        powers = 1 << np.arange(n, dtype=np.int64)
        row_support_masks = (support_bool.astype(np.int64) * powers).sum(axis=1)
        # Convert to Python ints so OR'ing across rows uses arbitrary-precision
        # arithmetic (n can exceed 63 in principle).
        row_support_masks = [int(m) for m in row_support_masks]

        first = self.tableau[0]
        tab_now = TableauPauliBasis(first.reshape(shape).copy())
        t_layers = [tab_now]
        layer_supports: list[int] = [row_support_masks[0]]

        add_new = False
        add_old = False
        # Add progress bar for layering
        for j in tqdm(
            range(1, self.stab_counts), desc="      Layering T-gates", leave=False
        ):
            tab_temp = self.tableau[j].reshape(shape).copy()
            new_support = row_support_masks[j]
            # Precompute the X<->Z-swapped row once; reused for every layer check below.
            swapped = self._swap_xz(tab_temp)
            # Determine how many layers to inspect from the end
            total_layers = len(t_layers)
            if max_layer_checks is None or max_layer_checks >= total_layers:
                lower_bound = 0
            else:
                lower_bound = total_layers - max_layer_checks

            insertion_done = False
            # Scan from the back towards the lower_bound
            for k in range(total_layers - 1, lower_bound - 1, -1):
                # Support-pruning: if the new Pauli has no qubit in common with
                # this layer's combined support, they trivially commute. Skip
                # the matmul.
                if not (new_support & layer_supports[k]):
                    continue
                tab = t_layers[k]
                tab_commute = tab._commutes_with_swapped(swapped)
                if not tab_commute:
                    if k == len(t_layers) - 1:
                        ## this means the current Pauli string do not commute with the latest layers, so need to create a new one.
                        add_new = True
                        insertion_done = True
                        break
                    else:
                        ## we find the earliest layer that does not commute with the current Pauli string. Add this pauli string to the next layer
                        t_layers[k + 1].append(tab_temp)
                        layer_supports[k + 1] |= new_support
                        add_old = True
                        insertion_done = True
                        break

            if add_new:
                tab_new = TableauPauliBasis(tab_temp)
                t_layers.append(tab_new)
                layer_supports.append(new_support)
                add_new = False
                continue

            if add_old:
                add_old = False
            else:
                # If we didn't finish insertion during the scan:
                # - If we scanned all layers (lower_bound==0) and all commute, place at layer 0 (earliest)
                # - Otherwise, place at the bounded earliest position (lower_bound)
                if not insertion_done:
                    target_index = 0 if lower_bound == 0 else lower_bound
                    t_layers[target_index].append(tab_temp)
                    layer_supports[target_index] |= new_support
                continue

        return t_layers
