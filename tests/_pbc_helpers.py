"""Helpers shared by the layering test suite.

Lives outside conftest.py so test files can import these symbols directly
(pytest's `tests` directory isn't a package).
"""

from __future__ import annotations

import numpy as np

from ftcircuitbench.pbc_converter.tab_gate import TableauPauliBasis


def pauli_string_to_row(pauli: str, sign: bool = False) -> np.ndarray:
    """Convert a Pauli string like 'XYZI' into a single tableau row.

    Layout: [X_0..X_{n-1}, Z_0..Z_{n-1}, sign].
    """
    n = len(pauli)
    row = np.zeros(2 * n + 1, dtype=bool)
    for i, ch in enumerate(pauli.upper()):
        if ch == "X":
            row[i] = True
        elif ch == "Z":
            row[n + i] = True
        elif ch == "Y":
            row[i] = True
            row[n + i] = True
        elif ch != "I":
            raise ValueError(f"Unexpected Pauli char {ch!r}")
    row[-1] = sign
    return row


def row_to_pauli_string(row: np.ndarray, qubits: int) -> str:
    """Inverse of pauli_string_to_row, ignoring sign."""
    out = []
    for i in range(qubits):
        x = bool(row[i])
        z = bool(row[qubits + i])
        out.append(
            {
                (False, False): "I",
                (True, False): "X",
                (False, True): "Z",
                (True, True): "Y",
            }[(x, z)]
        )
    return "".join(out)


def row_sign(row: np.ndarray) -> bool:
    """Extract the sign bit from a tableau row."""
    return bool(row[-1])


def make_tableau(
    paulis: list[str], signs: list[bool] | None = None
) -> TableauPauliBasis:
    """Build a TableauPauliBasis from a list of Pauli strings of equal length."""
    if signs is None:
        signs = [False] * len(paulis)
    rows = np.stack([pauli_string_to_row(p, s) for p, s in zip(paulis, signs)])
    return TableauPauliBasis(rows)


def paulis_in_layer(layer: TableauPauliBasis) -> list[str]:
    """Return Pauli strings (no sign) for every row in a layer."""
    return [
        row_to_pauli_string(layer.tableau[i], layer.qubits)
        for i in range(layer.stab_counts)
    ]


def paulis_with_signs_in_layer(layer: TableauPauliBasis) -> list[tuple[str, bool]]:
    """Return (pauli_string, sign) tuples for every row in a layer."""
    return [
        (
            row_to_pauli_string(layer.tableau[i], layer.qubits),
            row_sign(layer.tableau[i]),
        )
        for i in range(layer.stab_counts)
    ]


def collect_paulis(layers: list[TableauPauliBasis]) -> list[str]:
    """Flatten all Pauli strings across layers into one list (no signs)."""
    out: list[str] = []
    for layer in layers:
        out.extend(paulis_in_layer(layer))
    return out


def collect_paulis_with_signs(
    layers: list[TableauPauliBasis],
) -> list[tuple[str, bool]]:
    """Flatten all (Pauli, sign) pairs across layers into one list."""
    out: list[tuple[str, bool]] = []
    for layer in layers:
        out.extend(paulis_with_signs_in_layer(layer))
    return out
