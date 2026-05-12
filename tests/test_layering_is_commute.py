"""Unit tests for the symplectic commutation primitive `TableauPauliBasis.is_commute`.

This is the bedrock check the layering algorithm depends on. Every refactor
that touches `is_commute` (or its callers) must keep these passing.
"""

from __future__ import annotations

import numpy as np
import pytest
from _pbc_helpers import make_tableau, pauli_string_to_row

from ftcircuitbench.pbc_converter.tab_gate import TableauPauliBasis


def _commute(layer: TableauPauliBasis, target_row) -> bool:
    """Wrap is_commute to coerce the numpy bool scalar return to a Python bool."""
    return bool(layer.is_commute(target_row))


# ---------- Single-qubit known answers ----------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("I", "I", True),
        ("I", "X", True),
        ("I", "Y", True),
        ("I", "Z", True),
        ("X", "X", True),
        ("Y", "Y", True),
        ("Z", "Z", True),
        ("X", "Z", False),
        ("X", "Y", False),
        ("Y", "Z", False),
        ("Z", "X", False),
        ("Y", "X", False),
        ("Z", "Y", False),
    ],
)
def test_single_qubit_pairs(a: str, b: str, expected: bool) -> None:
    assert _commute(make_tableau([a]), pauli_string_to_row(b)) is expected


# ---------- Two-qubit known answers ----------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("XX", "ZZ", True),  # even overlap of anti-commuting pairs
        ("XX", "ZI", False),  # odd overlap
        ("XX", "IZ", False),  # odd overlap
        ("XY", "YX", True),  # q0 anti, q1 anti → commute
        ("XZ", "ZX", True),
        ("XI", "IX", True),  # disjoint support
        ("XI", "ZI", False),  # same qubit, anti
        ("YY", "XX", True),  # both pairs anti, count even
        ("YY", "XI", False),  # one pair anti
        ("YZ", "XY", True),  # q0 YX anti, q1 ZY anti → commute
    ],
)
def test_two_qubit_pairs(a: str, b: str, expected: bool) -> None:
    assert _commute(make_tableau([a]), pauli_string_to_row(b)) is expected


# ---------- Three-qubit spot checks ----------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("XYZ", "ZYX", True),  # q0 anti, q1 commute, q2 anti → commute
        ("XYZ", "ZIX", True),  # q0 anti, q1 commute, q2 anti → commute
        ("XXX", "YYY", False),  # 3 anti → anti-commute
        ("XII", "ZII", False),
        ("XYZ", "XYZ", True),  # identical
    ],
)
def test_three_qubit_pairs(a: str, b: str, expected: bool) -> None:
    assert _commute(make_tableau([a]), pauli_string_to_row(b)) is expected


# ---------- Group-vs-group: a tableau commutes iff every row commutes ----------


def test_commutes_with_every_row_in_layer() -> None:
    layer = make_tableau(["ZI", "IZ"])
    assert _commute(layer, pauli_string_to_row("ZZ")) is True
    # XI anti-commutes with ZI
    assert _commute(layer, pauli_string_to_row("XI")) is False


def test_anti_commute_with_any_row_means_overall_anti() -> None:
    layer = make_tableau(["ZZ", "XX"])
    # ZI anti-commutes with XX only (overall anti)
    assert _commute(layer, pauli_string_to_row("ZI")) is False
    # XZ anti-commutes with both ZZ and XX (overall anti)
    assert _commute(layer, pauli_string_to_row("XZ")) is False
    # YY commutes with both (2 anti-commutations each → even)
    assert _commute(layer, pauli_string_to_row("YY")) is True


# ---------- commutation_out=True returns the per-pair matrix ----------


def test_commutation_out_all_zero_when_commute() -> None:
    layer = make_tableau(["ZI", "IZ"])
    overall, matrix = layer.is_commute(pauli_string_to_row("ZZ"), commutation_out=True)
    assert bool(overall) is True
    assert matrix.shape == (1, 2)
    assert np.array_equal(matrix, np.array([[0, 0]]))


def test_commutation_out_flags_each_anti() -> None:
    layer = make_tableau(["ZI", "IZ"])
    overall, matrix = layer.is_commute(pauli_string_to_row("XI"), commutation_out=True)
    assert bool(overall) is False
    assert matrix.shape == (1, 2)
    # XI anti-commutes with ZI (col 0), commutes with IZ (col 1)
    assert matrix[0, 0] == 1
    assert matrix[0, 1] == 0


# ---------- Multi-row input tableau against multi-row layer ----------


def test_input_can_be_multi_row_tableau() -> None:
    layer = make_tableau(["ZZ"])
    candidates = make_tableau(["XX", "XI", "ZI"])
    overall, matrix = layer.is_commute(candidates, commutation_out=True)
    # XI anti-commutes with ZZ → overall False
    assert bool(overall) is False
    assert matrix.shape == (3, 1)
    assert matrix[0, 0] == 0  # XX commutes
    assert matrix[1, 0] == 1  # XI anti-commutes
    assert matrix[2, 0] == 0  # ZI commutes


# ---------- Error: qubit count mismatch ----------


def test_qubit_mismatch_raises() -> None:
    layer = make_tableau(["XX"])
    bad = pauli_string_to_row("XXX")
    with pytest.raises(ValueError):
        layer.is_commute(bad)


# ---------- Sign bit must not affect commutation ----------


def test_sign_bit_does_not_change_commutation() -> None:
    pos = make_tableau(["XX"], signs=[False])
    neg = make_tableau(["XX"], signs=[True])
    for target_str in ["ZZ", "ZI", "YY", "II"]:
        target = pauli_string_to_row(target_str)
        assert _commute(pos, target) == _commute(neg, target)


# ---------- Identity row is universal ----------


def test_identity_row_commutes_with_everything() -> None:
    n = 3
    identity_row = np.zeros(2 * n + 1, dtype=bool)
    layer = TableauPauliBasis(identity_row.reshape(1, -1).copy())
    for p in ["XYZ", "ZIX", "YYY", "III"]:
        assert _commute(layer, pauli_string_to_row(p)) is True
