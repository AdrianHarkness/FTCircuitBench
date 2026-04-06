from __future__ import annotations

from qiskit import QuantumCircuit

import ftcircuitbench.transpilers as transpilers_mod


def test_fast_wrapper_uses_cpp_when_available(monkeypatch) -> None:
    called = {"cpp": None}

    def _fake_cpp(*args, **kwargs):
        called["cpp"] = kwargs
        return QuantumCircuit(1)

    monkeypatch.setattr(transpilers_mod, "_is_nwqec_available", lambda: True)
    monkeypatch.setattr(transpilers_mod, "_nwqec_ct_transpiler", _fake_cpp)

    out = transpilers_mod.transpile_to_clifford_t_fast(
        QuantumCircuit(1),
        epsilon=1e-5,
        prefer_cpp=True,
        force_python=False,
    )
    assert isinstance(out, QuantumCircuit)
    assert called["cpp"]["epsilon"] == 1e-5


def test_fast_wrapper_converts_epsilon_to_precision_for_python(monkeypatch) -> None:
    called = {"python": None}

    def _fake_python(**kwargs):
        called["python"] = kwargs
        return QuantumCircuit(1)

    monkeypatch.setattr(transpilers_mod, "_is_nwqec_available", lambda: False)
    monkeypatch.setattr(transpilers_mod, "_python_gs_transpiler", _fake_python)

    out = transpilers_mod.transpile_to_clifford_t_fast(
        QuantumCircuit(1),
        epsilon=1e-4,
        prefer_cpp=True,
    )
    assert isinstance(out, QuantumCircuit)
    assert called["python"]["gridsynth_precision"] == 4
