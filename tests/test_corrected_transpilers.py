from __future__ import annotations

from qiskit import QuantumCircuit

import ftcircuitbench.transpilers as transpilers_mod


def test_gridsynth_wrapper_prefers_cpp_when_available(monkeypatch) -> None:
    calls = {"cpp": None}

    def _fake_cpp(**kwargs):
        calls["cpp"] = kwargs
        return QuantumCircuit(1)

    monkeypatch.setattr(transpilers_mod, "_is_nwqec_available", lambda: True)
    monkeypatch.setattr(transpilers_mod, "_nwqec_ct_transpiler", _fake_cpp)

    qc = QuantumCircuit(1)
    out = transpilers_mod.transpile_to_gridsynth_clifford_t(
        qc,
        gridsynth_precision=4,
        force_python=False,
        prefer_cpp=True,
    )
    assert isinstance(out, QuantumCircuit)
    assert calls["cpp"]["epsilon"] == 1e-4


def test_gridsynth_wrapper_honors_force_python(monkeypatch) -> None:
    calls = {"python": None}

    def _fake_python(**kwargs):
        calls["python"] = kwargs
        return QuantumCircuit(1)

    monkeypatch.setattr(transpilers_mod, "_is_nwqec_available", lambda: True)
    monkeypatch.setattr(transpilers_mod, "_python_gs_transpiler", _fake_python)

    qc = QuantumCircuit(1)
    out = transpilers_mod.transpile_to_gridsynth_clifford_t(
        qc,
        force_python=True,
        gridsynth_precision=2,
    )
    assert isinstance(out, QuantumCircuit)
    assert calls["python"]["gridsynth_precision"] == 2


def test_sk_wrapper_always_routes_to_python(monkeypatch) -> None:
    called = {"count": 0}

    def _fake_sk(**kwargs):
        called["count"] += 1
        return QuantumCircuit(1)

    monkeypatch.setattr(transpilers_mod, "_python_sk_transpiler", _fake_sk)
    out = transpilers_mod.transpile_to_solovay_kitaev_clifford_t(QuantumCircuit(1))
    assert isinstance(out, QuantumCircuit)
    assert called["count"] == 1
