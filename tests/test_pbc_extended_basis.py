"""
End-to-end tests that the PBC converter accepts the broader Clifford basis
{cx, h, s, sdg, x, y, z} as input (in addition to the T-layer gates {t, tdg}).

The converter previously only accepted {cx, h, s} natively (sdg was special-cased
to 3*S; x/y/z were rejected). After extending the tableau ops, the converter
should accept all of them and produce the same final state as the corresponding
manual decomposition.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from ftcircuitbench.pbc_converter.r_pauli_circ import RotationPauliCirc


def _process(qc: QuantumCircuit) -> RotationPauliCirc:
    rpc = RotationPauliCirc(qc)
    err = rpc.process(ifprint=False)
    assert err is False, f"PBC converter rejected circuit: {qc}"
    return rpc


def _measure_tableaus_equal(a: RotationPauliCirc, b: RotationPauliCirc) -> bool:
    return np.array_equal(a.measure_tab.tableau, b.measure_tab.tableau)


def _t_tableaus_equal(a: RotationPauliCirc, b: RotationPauliCirc) -> bool:
    return np.array_equal(a.t_tab.tableau, b.t_tab.tableau)


# ---------------------------------------------------------------------------
# Per-gate acceptance tests
# ---------------------------------------------------------------------------


def test_pbc_accepts_sdg_natively() -> None:
    qc = QuantumCircuit(1)
    qc.sdg(0)
    qc.t(0)
    _process(qc)


def test_pbc_accepts_x_natively() -> None:
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.t(0)
    _process(qc)


def test_pbc_accepts_y_natively() -> None:
    qc = QuantumCircuit(1)
    qc.y(0)
    qc.t(0)
    _process(qc)


def test_pbc_accepts_z_natively() -> None:
    qc = QuantumCircuit(1)
    qc.z(0)
    qc.t(0)
    _process(qc)


def test_pbc_accepts_combined_extended_basis() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.t(0)
    qc.tdg(1)
    qc.sdg(0)
    qc.x(0)
    qc.y(1)
    qc.z(0)
    qc.cx(0, 1)
    qc.t(1)
    _process(qc)


# ---------------------------------------------------------------------------
# Equivalence tests: native gate vs known Clifford decomposition
# ---------------------------------------------------------------------------


def test_native_sdg_equivalent_to_three_s() -> None:
    """PBC processing with sdg should yield the same tableau as 3*s."""
    qc_native = QuantumCircuit(2)
    qc_native.sdg(0)
    qc_native.t(0)
    qc_native.cx(0, 1)
    qc_native.t(1)

    qc_decomp = QuantumCircuit(2)
    qc_decomp.s(0)
    qc_decomp.s(0)
    qc_decomp.s(0)
    qc_decomp.t(0)
    qc_decomp.cx(0, 1)
    qc_decomp.t(1)

    a = _process(qc_native)
    b = _process(qc_decomp)
    assert _measure_tableaus_equal(a, b)
    assert _t_tableaus_equal(a, b)


def test_native_z_equivalent_to_two_s() -> None:
    """Z = S * S."""
    qc_native = QuantumCircuit(1)
    qc_native.z(0)
    qc_native.t(0)

    qc_decomp = QuantumCircuit(1)
    qc_decomp.s(0)
    qc_decomp.s(0)
    qc_decomp.t(0)

    a = _process(qc_native)
    b = _process(qc_decomp)
    assert _measure_tableaus_equal(a, b)
    assert _t_tableaus_equal(a, b)


def test_native_x_equivalent_to_h_z_h() -> None:
    """X = H Z H = H S S H."""
    qc_native = QuantumCircuit(1)
    qc_native.x(0)
    qc_native.t(0)

    qc_decomp = QuantumCircuit(1)
    qc_decomp.h(0)
    qc_decomp.s(0)
    qc_decomp.s(0)
    qc_decomp.h(0)
    qc_decomp.t(0)

    a = _process(qc_native)
    b = _process(qc_decomp)
    assert _measure_tableaus_equal(a, b)
    assert _t_tableaus_equal(a, b)


def test_native_y_equivalent_to_z_then_x() -> None:
    """Y = i*X*Z (the global phase factor doesn't appear in the tableau, only
    the relative Pauli matters; Y as a Clifford acts the same as ZX up to phase)."""
    qc_native = QuantumCircuit(1)
    qc_native.y(0)
    qc_native.t(0)

    qc_decomp = QuantumCircuit(1)
    qc_decomp.z(0)
    qc_decomp.x(0)
    qc_decomp.t(0)

    a = _process(qc_native)
    b = _process(qc_decomp)
    assert _measure_tableaus_equal(a, b)
    assert _t_tableaus_equal(a, b)
