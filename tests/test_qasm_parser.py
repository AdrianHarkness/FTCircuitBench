from __future__ import annotations

import pytest
from qiskit.quantum_info import Operator, process_fidelity

from ftcircuitbench.parser.qasm_parser import (
    _detect_qasm_version,
    load_qasm_circuit,
    rebuild_circuit_with_single_qreg,
    transpile_qasm_to_target_basis,
)


@pytest.mark.parametrize(
    ("qasm_text", "expected"),
    [
        ("OPENQASM 2.0;\nqreg q[1];", "2.0"),
        ("OPENQASM 3.0;\nqubit[1] q;", "3.0"),
        ("// no header\nqreg q[1];", "2.0"),
    ],
)
def test_detect_qasm_version(qasm_text: str, expected: str) -> None:
    assert _detect_qasm_version(qasm_text) == expected


def test_load_qasm2_string_removes_resets(qasm2_with_reset: str) -> None:
    qc = load_qasm_circuit(qasm2_with_reset, is_file=False)
    assert "reset" not in qc.count_ops()
    assert qc.num_qubits == 2
    assert qc.num_clbits == 2


def test_load_qasm3_string_removes_resets(qasm3_with_reset: str) -> None:
    pytest.importorskip("qiskit_qasm3_import")
    qc = load_qasm_circuit(qasm3_with_reset, is_file=False)
    assert "reset" not in qc.count_ops()
    assert qc.num_qubits == 2
    assert qc.num_clbits == 2


def test_load_qasm3_file_removes_resets(write_qasm, qasm3_with_reset: str) -> None:
    pytest.importorskip("qiskit_qasm3_import")
    qasm_file = write_qasm("with_reset.qasm", qasm3_with_reset)
    qc = load_qasm_circuit(str(qasm_file), is_file=True)
    assert "reset" not in qc.count_ops()


def test_load_qasm_file_not_found_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_qasm_circuit("/tmp/nonexistent_ftcircuitbench_file.qasm", is_file=True)


def test_rebuild_single_qreg_preserves_unitary(multi_register_circuit) -> None:
    source = multi_register_circuit.copy()
    source.remove_final_measurements(inplace=True)
    rebuilt = rebuild_circuit_with_single_qreg(source)

    assert len(rebuilt.qregs) == 1
    assert rebuilt.num_qubits == source.num_qubits
    assert rebuilt.num_clbits == source.num_clbits

    fid = process_fidelity(Operator(rebuilt), Operator(source))
    assert fid == pytest.approx(1.0, abs=1e-12)


def test_transpile_qasm_to_target_basis_restricts_gates(qasm2_with_reset: str) -> None:
    transpiled = transpile_qasm_to_target_basis(
        qasm2_with_reset,
        is_file=False,
        basis_gates=["rz", "h", "s", "cx"],
    )
    allowed = {"rz", "h", "s", "cx", "measure"}
    assert set(transpiled.count_ops()).issubset(allowed)
