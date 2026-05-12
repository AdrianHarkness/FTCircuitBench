"""Property-style invariants for the layering methods.

For every layering output, the following must hold:
  1. Multiset preservation: every input row appears in exactly one output layer.
  2. Within-layer commutation: every pair of rows in a layer commutes.
  3. Method-specific shape: singleton ⇒ N layers of 1.
  4. Edge cases: single rotation, all-commuting set, all-anti-commuting set.
"""

from __future__ import annotations

import itertools
from collections import Counter

import pytest
from _pbc_helpers import (
    collect_paulis_with_signs,
    make_tableau,
    paulis_in_layer,
)

from ftcircuitbench.pbc_converter.tab_gate import TableauPauliBasis

LAYER_METHODS = ["bare", "v2"]


def _run_method(tab: TableauPauliBasis, method: str):
    if method == "bare":
        return tab.layer()
    if method == "v2":
        return tab.layer_v2()
    if method == "singleton":
        return [
            TableauPauliBasis(tab.tableau[i].reshape(1, -1).copy())
            for i in range(tab.stab_counts)
        ]
    raise ValueError(method)


def _all_pairs_commute_within(layer: TableauPauliBasis) -> bool:
    """Brute-force pairwise check: every two rows in the layer commute."""
    rows = [layer.tableau[i] for i in range(layer.stab_counts)]
    for r1, r2 in itertools.combinations(rows, 2):
        # Build a single-row tableau from r1 and check against r2
        single = TableauPauliBasis(r1.reshape(1, -1).copy())
        if not bool(single.is_commute(r2)):
            return False
    return True


# ---------- Multiset preservation across all methods ----------


@pytest.mark.parametrize("method", LAYER_METHODS + ["singleton"])
def test_every_input_row_appears_exactly_once(method: str) -> None:
    paulis = ["XII", "IZI", "IIZ", "XZI", "ZIX", "YYI", "IXY", "ZZZ"]
    signs = [False, True, False, False, True, False, False, True]
    tab = make_tableau(paulis, signs=signs)
    layers = _run_method(tab, method)

    expected = Counter(zip(paulis, signs))
    got = Counter(collect_paulis_with_signs(layers))
    assert got == expected


# ---------- Within-layer commutation ----------


@pytest.mark.parametrize("method", LAYER_METHODS + ["singleton"])
def test_within_layer_commutation(method: str) -> None:
    paulis = ["XII", "IZI", "IIZ", "XZI", "ZIX", "YYI", "IXY", "ZZZ", "XXX", "YII"]
    tab = make_tableau(paulis)
    layers = _run_method(tab, method)
    for i, layer in enumerate(layers):
        assert _all_pairs_commute_within(layer), (
            f"method={method}, layer {i} = {paulis_in_layer(layer)} "
            "contains an anti-commuting pair"
        )


# ---------- Method-specific shape ----------


def test_singleton_produces_one_per_layer() -> None:
    paulis = ["XII", "IZI", "IIZ", "XZI"]
    tab = make_tableau(paulis)
    layers = _run_method(tab, "singleton")
    assert len(layers) == len(paulis)
    for layer in layers:
        assert layer.stab_counts == 1


# ---------- Edge cases ----------


@pytest.mark.parametrize("method", LAYER_METHODS + ["singleton"])
def test_single_rotation(method: str) -> None:
    tab = make_tableau(["XII"])
    layers = _run_method(tab, method)
    assert len(layers) == 1
    assert layers[0].stab_counts == 1
    assert paulis_in_layer(layers[0]) == ["XII"]


@pytest.mark.parametrize("method", ["bare", "v2"])
def test_all_commuting_set_collapses_to_one_layer(method: str) -> None:
    """Z_0, Z_1, Z_2 all commute pairwise → should fit in a single layer."""
    paulis = ["ZII", "IZI", "IIZ"]
    tab = make_tableau(paulis)
    layers = _run_method(tab, method)
    assert len(layers) == 1
    assert layers[0].stab_counts == 3


@pytest.mark.parametrize("method", LAYER_METHODS)
def test_all_anti_commuting_set_produces_n_layers(method: str) -> None:
    """X_0, Z_0, X_0 (interleaved anti-commuting) — each must end up in its own layer.

    We use rotations on a single qubit alternating X and Z so each consecutive pair
    anti-commutes.
    """
    paulis = ["X", "Z", "X", "Z"]
    tab = make_tableau(paulis)
    layers = _run_method(tab, method)
    # Each rotation should be in its own layer (consecutive rotations all anti-commute)
    assert len(layers) == 4
    for layer in layers:
        assert layer.stab_counts == 1


# ---------- Order preservation (relative order across layers) ----------


@pytest.mark.parametrize("method", LAYER_METHODS)
def test_relative_order_preserved_across_layers(method: str) -> None:
    """For order-preserving methods, the first appearance of each rotation respects input order.

    Concretely: if we tag rotations uniquely and read them off in (layer_index, position) order,
    the sequence of original input indices must be a valid topological ordering of the
    anti-commutation DAG. A simpler sufficient property we check here: for any two rotations
    that mutually anti-commute, their order in the output equals their order in the input.
    """
    paulis = ["X", "Z", "Y", "X", "Z"]  # all on 1 qubit, lots of anti-commutations
    tab = make_tableau(paulis)
    layers = _run_method(tab, method)

    # Walk layers in order, recording (input_index, pauli) for each placement
    # Match each output occurrence back to an input index by consuming the input list left-to-right
    remaining = list(enumerate(paulis))  # [(0,'X'), (1,'Z'), ...]
    output_order = []
    for layer in layers:
        for p in paulis_in_layer(layer):
            for idx, (orig_i, orig_p) in enumerate(remaining):
                if orig_p == p:
                    output_order.append(orig_i)
                    remaining.pop(idx)
                    break

    # Among output_order positions where the underlying Paulis anti-commute,
    # the order must match input order. For 1-qubit X/Y/Z, all distinct pairs anti-commute.
    for i in range(len(output_order)):
        for j in range(i + 1, len(output_order)):
            a_in = output_order[i]
            b_in = output_order[j]
            if paulis[a_in] != paulis[b_in]:  # distinct ⇒ anti-commute (1-qubit case)
                assert a_in < b_in, (
                    f"method={method}: rotation {a_in} ({paulis[a_in]}) appears after "
                    f"rotation {b_in} ({paulis[b_in]}) but they anti-commute"
                )


# ---------- max_layer_checks bound is respected (loose check) ----------


def test_v2_bounded_creates_at_least_as_many_layers_as_unbounded() -> None:
    """A tightly bounded v2 cannot insert into earlier layers, so it should produce
    at least as many layers as the unbounded version.
    """
    # Construct a sequence where unbounded v2 can collapse rotations into earlier
    # layers but v2 with max_layer_checks=1 cannot.
    paulis = [
        "ZII",
        "IZI",
        "XII",
        "IZI",
    ]  # XII anti-commutes with ZII; IZI commutes with both
    tab_unbounded = make_tableau(paulis)
    tab_bounded = make_tableau(paulis)
    layers_unbounded = tab_unbounded.layer_v2()
    layers_bounded = tab_bounded.layer_v2(max_layer_checks=1)
    assert len(layers_bounded) >= len(layers_unbounded)


# ---------- Sign bits flow through unchanged ----------


@pytest.mark.parametrize("method", LAYER_METHODS + ["singleton"])
def test_sign_bits_preserved(method: str) -> None:
    paulis = ["X", "Z", "Y", "X"]
    signs = [True, False, True, False]
    tab = make_tableau(paulis, signs=signs)
    layers = _run_method(tab, method)
    got = Counter(collect_paulis_with_signs(layers))
    assert got == Counter(zip(paulis, signs))
