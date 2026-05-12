"""
Unit tests for the native Pauli/Sdg ops on TableauForGate.

Three layers:
  1. Direct conjugation table (one-row tableau, each Pauli, expected (x, z, phase)).
  2. Algebraic identities (X^2 = I, Sdg = S^3, S * Sdg = I, etc).
  3. Cross-check against numpy-computed conjugation G * P * G^dagger.
"""

from __future__ import annotations

import numpy as np
import pytest

from ftcircuitbench.pbc_converter.tab_gate import TableauForGate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Single-qubit Pauli matrices.
I_MAT = np.array([[1, 0], [0, 1]], dtype=complex)
X_MAT = np.array([[0, 1], [1, 0]], dtype=complex)
Y_MAT = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_MAT = np.array([[1, 0], [0, -1]], dtype=complex)
H_MAT = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S_MAT = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG_MAT = np.array([[1, 0], [0, -1j]], dtype=complex)

# Tableau row encoding: (-1)^p * i^(x*z) * X^x * Z^z. The four single-qubit Paulis:
#   I=(0,0,0), X=(1,0,0), Z=(0,1,0), Y=(1,1,0).
PAULI_ROWS = {
    "I": np.array([[False, False, False]]),
    "X": np.array([[True, False, False]]),
    "Z": np.array([[False, True, False]]),
    "Y": np.array([[True, True, False]]),
}
PAULI_MATRICES = {"I": I_MAT, "X": X_MAT, "Y": Y_MAT, "Z": Z_MAT}

GATE_MATRICES = {
    "sdg": SDG_MAT,
    "x": X_MAT,
    "y": Y_MAT,
    "z": Z_MAT,
    "s": S_MAT,
    "h": H_MAT,
}


def _row_to_matrix(row: np.ndarray) -> np.ndarray:
    """Decode a single-qubit tableau row [x, z, p] into the represented Pauli matrix."""
    x, z, p = bool(row[0]), bool(row[1]), bool(row[2])
    base = np.linalg.matrix_power(X_MAT, int(x)) @ np.linalg.matrix_power(Z_MAT, int(z))
    factor = ((-1) ** int(p)) * (1j ** (int(x) * int(z)))
    return factor * base


def _matrices_equal(a: np.ndarray, b: np.ndarray, atol: float = 1e-10) -> bool:
    return bool(np.allclose(a, b, atol=atol))


def _make_one_qubit_tab(row: np.ndarray) -> TableauForGate:
    return TableauForGate(row.copy())


# ---------------------------------------------------------------------------
# Layer 1: direct conjugation table for each new gate
# ---------------------------------------------------------------------------

# Expected (x, z, phase) outputs after applying gate to each input Pauli.
# Derived analytically from G P G^dagger.
EXPECTED = {
    "sdg": {
        "I": (False, False, False),
        "X": (True, True, True),  # Sdg X Sdg^dag = -Y
        "Y": (True, False, False),  # Sdg Y Sdg^dag = X
        "Z": (False, True, False),  # Sdg Z Sdg^dag = Z
    },
    "x": {
        "I": (False, False, False),
        "X": (True, False, False),
        "Y": (True, True, True),  # X Y X = -Y
        "Z": (False, True, True),  # X Z X = -Z
    },
    "y": {
        "I": (False, False, False),
        "X": (True, False, True),  # Y X Y = -X
        "Y": (True, True, False),
        "Z": (False, True, True),  # Y Z Y = -Z
    },
    "z": {
        "I": (False, False, False),
        "X": (True, False, True),  # Z X Z = -X
        "Y": (True, True, True),  # Z Y Z = -Y
        "Z": (False, True, False),
    },
}


@pytest.mark.parametrize("gate", ["sdg", "x", "y", "z"])
@pytest.mark.parametrize("pauli", ["I", "X", "Y", "Z"])
def test_layer1_conjugation_table(gate: str, pauli: str) -> None:
    tab = _make_one_qubit_tab(PAULI_ROWS[pauli])
    tab.apply_gate(gate, [0])
    got = (bool(tab.tableau[0, 0]), bool(tab.tableau[0, 1]), bool(tab.tableau[0, 2]))
    assert (
        got == EXPECTED[gate][pauli]
    ), f"{gate} on {pauli}: got {got}, expected {EXPECTED[gate][pauli]}"


# ---------------------------------------------------------------------------
# Layer 2: algebraic identities (sequence comparisons)
# ---------------------------------------------------------------------------


def _all_pauli_rows_one_qubit() -> np.ndarray:
    """Tableau holding all four single-qubit Paulis as separate rows."""
    return np.array(
        [
            [False, False, False],  # I
            [True, False, False],  # X
            [False, True, False],  # Z
            [True, True, False],  # Y
        ]
    )


def _apply_sequence(rows: np.ndarray, ops: list[tuple[str, list[int]]]) -> np.ndarray:
    tab = TableauForGate(rows.copy())
    for name, qs in ops:
        tab.apply_gate(name, qs)
    return tab.tableau.copy()


def test_layer2_x_squared_is_identity() -> None:
    start = _all_pauli_rows_one_qubit()
    end = _apply_sequence(start, [("x", [0]), ("x", [0])])
    assert np.array_equal(end, start)


def test_layer2_y_squared_is_identity() -> None:
    start = _all_pauli_rows_one_qubit()
    end = _apply_sequence(start, [("y", [0]), ("y", [0])])
    assert np.array_equal(end, start)


def test_layer2_z_squared_is_identity() -> None:
    start = _all_pauli_rows_one_qubit()
    end = _apply_sequence(start, [("z", [0]), ("z", [0])])
    assert np.array_equal(end, start)


def test_layer2_s_then_sdg_is_identity() -> None:
    start = _all_pauli_rows_one_qubit()
    end = _apply_sequence(start, [("s", [0]), ("sdg", [0])])
    assert np.array_equal(end, start)


def test_layer2_sdg_then_s_is_identity() -> None:
    start = _all_pauli_rows_one_qubit()
    end = _apply_sequence(start, [("sdg", [0]), ("s", [0])])
    assert np.array_equal(end, start)


def test_layer2_sdg_equals_three_s() -> None:
    start = _all_pauli_rows_one_qubit()
    via_sdg = _apply_sequence(start, [("sdg", [0])])
    via_three_s = _apply_sequence(start, [("s", [0]), ("s", [0]), ("s", [0])])
    assert np.array_equal(via_sdg, via_three_s)


def test_layer2_sdg_squared_equals_z() -> None:
    """Sdg^2 = (S^-1)^2 = (S^2)^-1 = Z^-1 = Z."""
    start = _all_pauli_rows_one_qubit()
    via_sdg_sq = _apply_sequence(start, [("sdg", [0]), ("sdg", [0])])
    via_z = _apply_sequence(start, [("z", [0])])
    assert np.array_equal(via_sdg_sq, via_z)


def test_layer2_z_equals_two_s() -> None:
    """Z = S^2."""
    start = _all_pauli_rows_one_qubit()
    via_z = _apply_sequence(start, [("z", [0])])
    via_ss = _apply_sequence(start, [("s", [0]), ("s", [0])])
    assert np.array_equal(via_z, via_ss)


def test_layer2_x_equals_h_z_h() -> None:
    """X = H Z H."""
    start = _all_pauli_rows_one_qubit()
    via_x = _apply_sequence(start, [("x", [0])])
    via_hzh = _apply_sequence(start, [("h", [0]), ("z", [0]), ("h", [0])])
    assert np.array_equal(via_x, via_hzh)


# ---------------------------------------------------------------------------
# Layer 3: cross-check against numpy-computed G P G^dagger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate", ["sdg", "x", "y", "z"])
@pytest.mark.parametrize("pauli", ["I", "X", "Y", "Z"])
def test_layer3_matches_explicit_conjugation(gate: str, pauli: str) -> None:
    G = GATE_MATRICES[gate]
    P = PAULI_MATRICES[pauli]
    expected = G @ P @ G.conj().T

    tab = _make_one_qubit_tab(PAULI_ROWS[pauli])
    tab.apply_gate(gate, [0])
    got = _row_to_matrix(tab.tableau[0])
    assert _matrices_equal(
        got, expected
    ), f"{gate} {pauli}: tableau row -> {got}, expected {expected}"


# ---------------------------------------------------------------------------
# Layer 3 (multi-qubit): verify gates only touch their target qubit's row bits
# ---------------------------------------------------------------------------


def test_apply_gate_only_touches_target_qubit_columns() -> None:
    """Apply each new gate to qubit 0 of a 2-qubit row and check qubit 1 is unchanged."""
    # Row encodes Y on qubit 0, X on qubit 1: x=[1,1], z=[1,0], phase=0 -> i*XZ on q0 * X on q1.
    row = np.array([[True, True, True, False, False]])  # 2*2+1 = 5 cols
    for gate in ["sdg", "x", "y", "z"]:
        tab = TableauForGate(row.copy())
        tab.apply_gate(gate, [0])
        # Qubit 1 columns (index 1 for X-bit, index 3 for Z-bit) untouched
        assert tab.tableau[0, 1] == row[0, 1]
        assert tab.tableau[0, 3] == row[0, 3]
