from __future__ import annotations

from pathlib import Path

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def qasm_dir(repo_root: Path) -> Path:
    return repo_root / "qasm"


@pytest.fixture
def write_qasm(tmp_path: Path):
    def _write(name: str, contents: str) -> Path:
        path = tmp_path / name
        path.write_text(contents, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def qasm2_with_reset() -> str:
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
reset q[0];
cx q[0],q[1];
measure q[1] -> c[1];
"""


@pytest.fixture
def qasm3_with_reset() -> str:
    return """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
reset q[0];
cx q[0], q[1];
c[1] = measure q[1];
"""


@pytest.fixture
def simple_two_qubit_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rz(0.3, 0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def multi_register_circuit() -> QuantumCircuit:
    qa = QuantumRegister(1, "a")
    qb = QuantumRegister(2, "b")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qa, qb, c)
    qc.h(qa[0])
    qc.cx(qa[0], qb[1])
    qc.s(qb[0])
    qc.measure(qa[0], c[0])
    return qc


@pytest.fixture
def large_rz_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(8)
    for i in range(8):
        qc.rz(0.1 * (i + 1), i)
    for i in range(7):
        qc.cx(i, i + 1)
    return qc
