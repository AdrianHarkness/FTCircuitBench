from __future__ import annotations

import random
from pathlib import Path

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from ftcircuitbench.pbc_converter.r_pauli_circ import RotationPauliCirc


@pytest.fixture
def small_clifford_t_circuit() -> QuantumCircuit:
    """A small deterministic Clifford+T circuit covering H/S/CX + several T/Tdg."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.t(0)
    qc.cx(0, 1)
    qc.t(1)
    qc.h(2)
    qc.tdg(2)
    qc.cx(1, 2)
    qc.t(0)
    qc.s(1)
    qc.t(2)
    return qc


@pytest.fixture
def random_clifford_t_circuit() -> QuantumCircuit:
    """A larger deterministic random Clifford+T circuit (seeded)."""
    rng = random.Random(20260508)
    n = 4
    qc = QuantumCircuit(n)
    gates_1q = ["h", "s", "t", "tdg"]
    for _ in range(40):
        if rng.random() < 0.3:
            a, b = rng.sample(range(n), 2)
            qc.cx(a, b)
        else:
            getattr(qc, rng.choice(gates_1q))(rng.randrange(n))
    return qc


@pytest.fixture
def make_rpc():
    """Factory: returns a processed RotationPauliCirc for a circuit."""

    def _make(qc: QuantumCircuit) -> RotationPauliCirc:
        rpc = RotationPauliCirc(qc)
        err = rpc.process(ifprint=False)
        assert err is False
        return rpc

    return _make


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
