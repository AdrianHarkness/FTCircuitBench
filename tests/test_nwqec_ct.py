from __future__ import annotations

import types

import pytest
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps

from ftcircuitbench.transpilers.nwqec_ct import (
    is_nwqec_available,
    transpile_to_clifford_t_cpp,
)


class _FakeNwqecCircuit:
    def __init__(self, qasm_out: str):
        self._qasm_out = qasm_out

    def to_qasm(self) -> str:
        return self._qasm_out


def _install_fake_nwqec(monkeypatch, with_cpp: bool = True):
    calls = {"kwargs": None}
    out_qc = QuantumCircuit(2)
    out_qc.h(0)
    out_qc.t(0)
    out_qc.cx(0, 1)
    out_qasm = qasm2_dumps(out_qc)

    def _load_qasm(_path: str):
        return _FakeNwqecCircuit(out_qasm)

    def _to_clifford_t(circ, keep_ccx=False, **kwargs):
        calls["kwargs"] = {"keep_ccx": keep_ccx, **kwargs}
        return circ

    fake = types.SimpleNamespace(
        WITH_GRIDSYNTH_CPP=with_cpp,
        load_qasm=_load_qasm,
        to_clifford_t=_to_clifford_t,
    )
    monkeypatch.setitem(__import__("sys").modules, "nwqec", fake)
    return calls


def test_is_nwqec_available_false_when_module_missing(monkeypatch) -> None:
    builtins_mod = __import__("builtins")
    original_import = builtins_mod.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "nwqec":
            raise ImportError("forced import failure for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins_mod, "__import__", _raising_import)
    assert is_nwqec_available() is False


def test_transpile_cpp_strict_mode_rejects_python_fallback(monkeypatch) -> None:
    _install_fake_nwqec(monkeypatch, with_cpp=False)
    qc = QuantumCircuit(1)
    with pytest.raises(RuntimeError, match="refusing Python fallback"):
        transpile_to_clifford_t_cpp(qc, forbid_python_fallback=True)


def test_transpile_cpp_returns_intermediate_and_output(monkeypatch) -> None:
    calls = _install_fake_nwqec(monkeypatch, with_cpp=True)
    qc = QuantumCircuit(2)
    qc.rz(0.3, 0)
    qc.cx(0, 1)

    inter, out = transpile_to_clifford_t_cpp(
        qc,
        epsilon=1e-3,
        return_intermediate=True,
        forbid_python_fallback=True,
    )
    assert isinstance(inter, QuantumCircuit)
    assert isinstance(out, QuantumCircuit)
    assert set(out.count_ops()).issubset({"cx", "h", "s", "t", "tdg"})
    assert calls["kwargs"]["epsilon"] == pytest.approx(1e-3)


def test_transpile_cpp_rejects_invalid_input_type(monkeypatch) -> None:
    _install_fake_nwqec(monkeypatch, with_cpp=True)
    with pytest.raises(TypeError, match="QuantumCircuit or str"):
        transpile_to_clifford_t_cpp(123, forbid_python_fallback=True)
