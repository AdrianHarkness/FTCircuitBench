# ./ftcircuitbench/pbc_converter/layers.py
"""
Recovery of PBC layer structure from an ordered sequence of Pauli operators.

A PBC "layer" is a maximal run of mutually commuting rotations that may be
applied together. The Python path computes layers explicitly
(`RotationPauliCirc.layering`) and marks them with barriers; the nwqec backend
emits a flat operator stream whose layer boundaries are implicit in the ordering
(`nwqec.fuse_t` groups commuting operators adjacently).

Scanning the emitted order for maximal mutually-commuting runs recovers those
boundaries exactly -- for Python output it reproduces `layering()`'s partition
element-for-element, and for fused nwqec output it recovers layers of the same
width. When no grouping has been performed the scan degenerates to singleton
layers, which is the honest answer: no layering exists to report.

Note this *reads off* structure already present in the ordering rather than
re-deriving an optimal layering; it never reorders operators.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .tab_gate import TableauForGate

__all__ = ["paulis_commute", "commuting_layer_runs"]


def _as_tableau(pauli_str: str) -> TableauForGate:
    """Convert a single signed/unsigned full-width Pauli string to a tableau."""
    return TableauForGate.convert_back(pauli_str)


def paulis_commute(pauli_a: str, pauli_b: str) -> bool:
    """Return True if two full-width Pauli strings commute.

    Accepts optional leading '+'/'-'; the phase does not affect commutation.

    >>> paulis_commute("+XX", "+ZZ")
    True
    >>> paulis_commute("+XI", "+ZI")
    False
    """
    return bool(_as_tableau(pauli_a).is_commute(_as_tableau(pauli_b)))


def commuting_layer_runs(pauli_strings: Sequence[str]) -> List[List[str]]:
    """Partition an ordered Pauli sequence into maximal mutually-commuting runs.

    Args:
        pauli_strings: Full-width Pauli strings ("+IXZI"), in emission order.
            All entries must describe the same number of qubits.

    Returns:
        A list of layers, each a list of the original strings in their original
        order. Concatenating the layers reproduces the input exactly.

    Raises:
        ValueError: If the strings do not all describe the same qubit count.
    """
    if not pauli_strings:
        return []

    widths = {len(p.lstrip("+-")) for p in pauli_strings}
    if len(widths) > 1:
        raise ValueError(
            f"All Pauli strings must cover the same number of qubits; got widths {sorted(widths)}"
        )

    layers: List[List[str]] = []
    current: List[str] = []
    current_rows: List[np.ndarray] = []

    for pauli in pauli_strings:
        tab = _as_tableau(pauli)
        if current:
            # One call tests the candidate against every operator already in the
            # layer: is_commute over a stacked tableau is True only when the
            # candidate commutes with all rows.
            layer_tab = TableauForGate(np.vstack(current_rows))
            if not bool(layer_tab.is_commute(tab)):
                layers.append(current)
                current = []
                current_rows = []
        current.append(pauli)
        current_rows.append(tab.tableau)

    if current:
        layers.append(current)
    return layers
