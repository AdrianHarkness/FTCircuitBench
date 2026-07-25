"""
PBC output-format contract tests.

Both PBC backends -- the Python tableau path (`convert_to_pbc_circuit` with
`use_nwqec=False`) and the nwqec C++ path -- must emit one canonical format:

  * rotations   ``R<pauli>(pi/8)`` / ``R<pauli>(-pi/8)``, Pauli compressed to the
    active (non-identity) qubits the gate binds to
  * measurements ``Meas<sign><pauli>`` -- the sign is always present, because
    +ZZ and -ZZ are different observables
  * T-layers separated by ``barrier``
  * the same ``*_tlayers.txt`` / ``*_measure_basis.txt`` artifacts on disk

These tests pin that contract; before they existed nothing asserted on PBC label
format, and the Python path silently dropped measurement signs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from ftcircuitbench.analyzer.pbc_analyzer import (
    analyze_pbc_circuit,
    parse_pbc_gate_name,
    split_pauli_sign,
)
from ftcircuitbench.pbc_converter.layers import commuting_layer_runs, paulis_commute
from ftcircuitbench.pbc_converter.nwqec_adapter import is_nwqec_available
from ftcircuitbench.pbc_converter.pbc_circuit_reader import read_combined_pbc_file
from ftcircuitbench.pbc_converter.pbc_generator import convert_to_pbc_circuit
from ftcircuitbench.pbc_converter.r_pauli_circ import RotationPauliCirc

ROTATION_RE = re.compile(r"^R[IXYZ]+\((-?pi/8)\)$")
MEASUREMENT_RE = re.compile(r"^Meas[+-][IXYZ]+$")


def _clifford_t_circuit() -> QuantumCircuit:
    """Small Clifford+T input; needs no gridsynth, so these tests always run."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.t(0)
    qc.cx(0, 1)
    qc.tdg(1)
    qc.h(2)
    qc.t(2)
    qc.cx(1, 2)
    return qc


def _requires_nwqec() -> None:
    pytest.importorskip("nwqec")
    if not is_nwqec_available():
        pytest.skip("nwqec not available")


def _pbc(use_nwqec: bool, output_prefix: str | None = None) -> QuantumCircuit:
    circuit, _stats = convert_to_pbc_circuit(
        _clifford_t_circuit(),
        if_print_rpc=False,
        use_nwqec=use_nwqec,
        output_prefix=output_prefix,
    )
    return circuit


# ---------------------------------------------------------------------------
# Gate-label grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_nwqec", [False, True], ids=["python", "nwqec"])
def test_measurement_labels_are_signed(use_nwqec: bool) -> None:
    """Regression: the Python path used to emit unsigned 'MeasXZ', losing the
    phase bit it had already computed."""
    if use_nwqec:
        _requires_nwqec()
    labels = [
        i.operation.name
        for i in _pbc(use_nwqec).data
        if i.operation.name.startswith("Meas")
    ]
    assert labels, "expected at least one measurement operator"
    for label in labels:
        assert MEASUREMENT_RE.match(label), f"unsigned or malformed label: {label!r}"


@pytest.mark.parametrize("use_nwqec", [False, True], ids=["python", "nwqec"])
def test_rotation_labels_match_grammar(use_nwqec: bool) -> None:
    if use_nwqec:
        _requires_nwqec()
    labels = [
        i.operation.name
        for i in _pbc(use_nwqec).data
        if i.operation.name.startswith("R")
    ]
    assert labels, "expected at least one rotation operator"
    for label in labels:
        assert ROTATION_RE.match(label), f"malformed rotation label: {label!r}"


def test_backends_agree_on_labels() -> None:
    """The two backends must produce the same label stream for the same input."""
    _requires_nwqec()
    py = [i.operation.name for i in _pbc(False).data]
    nw = [i.operation.name for i in _pbc(True).data]
    assert py == nw


@pytest.mark.parametrize("use_nwqec", [False, True], ids=["python", "nwqec"])
def test_layers_are_barrier_separated(use_nwqec: bool) -> None:
    if use_nwqec:
        _requires_nwqec()
    names = [i.operation.name for i in _pbc(use_nwqec).data]
    assert "barrier" in names, "expected barriers delimiting T-layers"
    assert names[0] != "barrier", "no leading barrier before the first layer"


# ---------------------------------------------------------------------------
# Pauli weight must be measured on the bare Pauli
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_nwqec", [False, True], ids=["python", "nwqec"])
def test_measurement_weight_matches_qubit_count(use_nwqec: bool) -> None:
    """Regression: the sign character was counted as a Pauli, inflating every
    measurement weight by one."""
    if use_nwqec:
        _requires_nwqec()
    circuit = _pbc(use_nwqec)
    seen = 0
    for instruction in circuit.data:
        op_type, pauli_in_name, _ = parse_pbc_gate_name(instruction.operation.name)
        if op_type != "measurement":
            continue
        seen += 1
        _sign, bare = split_pauli_sign(pauli_in_name)
        assert len(bare) == len(instruction.qubits), (
            f"weight {len(bare)} from {pauli_in_name!r} != "
            f"{len(instruction.qubits)} bound qubits"
        )
    assert seen, "expected at least one measurement operator"


@pytest.mark.parametrize(
    "label,expected_sign,expected_bare",
    [
        ("MeasXZ", "+", "XZ"),
        ("Meas+XZ", "+", "XZ"),
        ("Meas-XZ", "-", "XZ"),
        ("MeasY", "+", "Y"),
    ],
)
def test_parse_and_split_round_trip(
    label: str, expected_sign: str, expected_bare: str
) -> None:
    """Signed and unsigned labels both reduce to the same bare Pauli."""
    op_type, pauli_in_name, params = parse_pbc_gate_name(label)
    assert op_type == "measurement"
    assert params is None
    sign, bare = split_pauli_sign(pauli_in_name)
    assert (sign, bare) == (expected_sign, expected_bare)


def test_barriers_do_not_count_as_pbc_operators() -> None:
    """Adding layer barriers must not inflate the operator totals."""
    circuit = _pbc(False)
    stats = analyze_pbc_circuit(circuit)
    n_rot = sum(1 for i in circuit.data if i.operation.name.startswith("R"))
    n_meas = sum(1 for i in circuit.data if i.operation.name.startswith("Meas"))
    assert stats["pbc_total_operators"] == n_rot + n_meas
    assert parse_pbc_gate_name("barrier")[0] == "utility"


# ---------------------------------------------------------------------------
# Artifact files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_nwqec", [False, True], ids=["python", "nwqec"])
def test_output_prefix_writes_all_artifacts(use_nwqec: bool, tmp_path: Path) -> None:
    """Regression: with nwqec active (the default) output_prefix was ignored and
    no files were written at all."""
    if use_nwqec:
        _requires_nwqec()
    prefix = tmp_path / "circ"
    _pbc(use_nwqec, output_prefix=str(prefix))

    expected = [
        "circ_pre_opt_tlayers.txt",
        "circ_pre_opt_measure_basis.txt",
        "circ_post_opt_tlayers.txt",
        "circ_post_opt_measure_basis.txt",
    ]
    for name in expected:
        path = tmp_path / name
        assert path.exists(), f"missing artifact {name}"
        assert path.stat().st_size > 0, f"empty artifact {name}"


@pytest.mark.parametrize("use_nwqec", [False, True], ids=["python", "nwqec"])
def test_artifacts_parse_under_existing_reader(use_nwqec: bool, tmp_path: Path) -> None:
    if use_nwqec:
        _requires_nwqec()
    prefix = tmp_path / "circ"
    _pbc(use_nwqec, output_prefix=str(prefix))

    parsed = read_combined_pbc_file(str(tmp_path / "circ_post_opt_tlayers.txt"))
    assert parsed["t_layers"], "reader found no T-layers"
    for layer in parsed["t_layers"]:
        for pauli in layer:
            assert pauli[0] in "+-", f"artifact Pauli lost its sign: {pauli!r}"


def test_artifact_files_identical_across_backends(tmp_path: Path) -> None:
    """Both backends must produce byte-identical artifacts for the same input."""
    _requires_nwqec()
    py_prefix = tmp_path / "py"
    nw_prefix = tmp_path / "nw"
    _pbc(False, output_prefix=str(py_prefix))
    _pbc(True, output_prefix=str(nw_prefix))

    for suffix in ("_post_opt_tlayers.txt", "_post_opt_measure_basis.txt"):
        py_text = (tmp_path / f"py{suffix}").read_text()
        nw_text = (tmp_path / f"nw{suffix}").read_text()
        assert py_text == nw_text, f"artifact mismatch in {suffix}"


# ---------------------------------------------------------------------------
# Layer recovery
# ---------------------------------------------------------------------------


def test_paulis_commute_basic() -> None:
    assert paulis_commute("+XX", "+ZZ")
    assert paulis_commute("+XI", "+IZ")
    assert not paulis_commute("+XI", "+ZI")
    # Phase must not affect commutation.
    assert paulis_commute("-XX", "+ZZ")


def test_commuting_layer_runs_edge_cases() -> None:
    assert commuting_layer_runs([]) == []
    assert commuting_layer_runs(["+XI"]) == [["+XI"]]
    # Anticommuting neighbours must land in separate layers.
    assert commuting_layer_runs(["+XI", "+ZI"]) == [["+XI"], ["+ZI"]]
    with pytest.raises(ValueError):
        commuting_layer_runs(["+XI", "+XYZ"])


@pytest.mark.parametrize("method", ["v2", "bare"])
def test_commuting_layer_runs_reproduces_layering(method: str) -> None:
    """The scan must recover exactly the partition RotationPauliCirc computed --
    this is what licenses using it to place barriers on the nwqec path."""
    rpc = RotationPauliCirc(_clifford_t_circuit())
    assert rpc.process(ifprint=False) is False
    rpc.layering(method=method, ifprint=False)

    truth = [
        [tab.readout(i) for i in range(tab.stab_counts)]
        for tab in rpc.t_layers
        if tab and tab.stab_counts > 0
    ]
    flat = [pauli for layer in truth for pauli in layer]
    recovered = commuting_layer_runs(flat)

    assert recovered == truth
    assert [p for layer in recovered for p in layer] == flat
