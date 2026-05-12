from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

import ftcircuitbench.decomposer.decomposer as decomposer_mod
import ftcircuitbench.fidelity as fidelity_mod
from ftcircuitbench.decomposer import decompose_rz_gates_gridsynth
from ftcircuitbench.fidelity import (
    calculate_circuit_fidelity,
    rz_product_fidelity,
    rz_product_fidelity_sk,
)


def test_calculate_fidelity_unitary_path_success(simple_two_qubit_circuit) -> None:
    result = calculate_circuit_fidelity(
        simple_two_qubit_circuit,
        simple_two_qubit_circuit.copy(),
        gridsynth_precision=3,
    )
    assert result["method"] == "unitary_based"
    assert result["status"] == "success"
    assert result["fidelity"] == pytest.approx(1.0, abs=1e-12)


def test_calculate_fidelity_unitary_path_error_status() -> None:
    original = QuantumCircuit(1)
    original.h(0)
    decomposed = QuantumCircuit(2)
    decomposed.cx(0, 1)

    result = calculate_circuit_fidelity(original, decomposed, gridsynth_precision=3)
    assert result["method"] == "unitary_based"
    assert str(result["status"]).startswith("error:")
    assert result["fidelity"] is None


def test_calculate_fidelity_large_without_intermediate_returns_na(
    large_rz_circuit,
) -> None:
    result = calculate_circuit_fidelity(
        large_rz_circuit,
        large_rz_circuit.copy(),
        gridsynth_precision=3,
        sk_recursion_degree=1,
        intermediate_qc=None,
    )
    assert result["fidelity"] == "N/A"
    assert result["status"] == "not_available_no_intermediate_circuit"
    assert result["method"] == "rz_product_fidelity_sk"


def test_calculate_fidelity_large_uses_sk_product(
    monkeypatch, large_rz_circuit
) -> None:
    monkeypatch.setattr(
        fidelity_mod,
        "rz_product_fidelity_sk",
        lambda _c, _r: {
            "overall_fidelity": 0.91,
            "individual_fidelities": [0.91],
            "rz_gate_count": 1,
            "status": "success",
        },
    )
    result = calculate_circuit_fidelity(
        large_rz_circuit,
        large_rz_circuit.copy(),
        gridsynth_precision=3,
        sk_recursion_degree=2,
        intermediate_qc=large_rz_circuit,
    )
    assert result["method"] == "rz_product_fidelity_sk"
    assert result["fidelity"] == pytest.approx(0.91)
    assert result["rz_gate_count"] == 1


def test_calculate_fidelity_large_uses_gridsynth_product(
    monkeypatch, large_rz_circuit
) -> None:
    monkeypatch.setattr(
        fidelity_mod,
        "rz_product_fidelity",
        lambda _c, _p: {
            "overall_fidelity": 0.88,
            "individual_fidelities": [0.95, 0.93],
            "rz_gate_count": 2,
            "status": "partial_failure",
        },
    )
    result = calculate_circuit_fidelity(
        large_rz_circuit,
        large_rz_circuit.copy(),
        gridsynth_precision=3,
        intermediate_qc=large_rz_circuit,
    )
    assert result["method"] == "rz_product_fidelity"
    assert result["fidelity"] == pytest.approx(0.88)
    assert result["status"] == "partial_failure"


def test_rz_product_fidelity_no_rz_gates_returns_contract() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["status"] == "no_rz_gates"
    assert result["overall_fidelity"] == "N/A"
    assert result["rz_gate_count"] == 0


def test_rz_product_fidelity_skips_parameterized_rz(monkeypatch) -> None:
    monkeypatch.setattr(
        fidelity_mod, "_run_gridsynth_cli", lambda *_args, **_kwargs: "T"
    )
    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    qc.rz(theta, 0)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["status"] == "no_rz_gates"
    assert result["rz_gate_count"] == 0


def test_rz_product_fidelity_sequential_success(monkeypatch) -> None:
    monkeypatch.setattr(
        fidelity_mod, "_run_gridsynth_cli", lambda *_args, **_kwargs: "T"
    )
    qc = QuantumCircuit(1)
    qc.rz(0.2, 0)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["status"] in {"success", "partial_failure"}
    assert result["rz_gate_count"] == 1
    assert isinstance(result["overall_fidelity"], float)
    assert result["multiprocessing_used"] is False


def test_rz_product_fidelity_multiprocessing_fallback(monkeypatch) -> None:
    class _BrokenPool:
        def __enter__(self):
            raise RuntimeError("pool setup failed")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        fidelity_mod.multiprocessing,
        "Pool",
        lambda *args, **kwargs: _BrokenPool(),
    )
    monkeypatch.setattr(
        fidelity_mod, "_run_gridsynth_cli", lambda *_args, **_kwargs: "W"
    )

    qc = QuantumCircuit(2)
    qc.rz(0.2, 0)
    qc.rz(0.3, 1)
    with pytest.warns(RuntimeWarning, match="multiprocessing failed"):
        result = rz_product_fidelity(
            qc, gridsynth_precision=3, use_multiprocessing=True
        )
    assert result["status"] == "success"
    assert result["rz_gate_count"] == 2
    assert result["multiprocessing_used"] is False


# ---------------------------------------------------------------------------
# Numerical invariants (currently not asserted anywhere)
# ---------------------------------------------------------------------------


def test_overall_fidelity_equals_product_of_individuals(monkeypatch) -> None:
    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", lambda *_a, **_k: "H")
    qc = QuantumCircuit(1)
    qc.rz(0.4, 0)
    qc.rz(0.7, 0)
    qc.rz(1.1, 0)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["overall_fidelity"] == pytest.approx(
        float(np.prod(result["individual_fidelities"]))
    )


def test_min_max_avg_match_individual_fidelities(monkeypatch) -> None:
    # Mix identity (f=1.0) and bad (f<1) decompositions to spread the values
    def _stub(theta_str, *_a, **_k):
        return "" if abs(float(theta_str)) < 0.5 else "H"

    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", _stub)
    qc = QuantumCircuit(1)
    qc.rz(0.1, 0)  # identity branch -> 1.0
    qc.rz(0.9, 0)  # H stub -> low
    qc.rz(1.4, 0)  # H stub -> low (different value)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    fids = result["individual_fidelities"]
    assert result["min_individual_fidelity"] == pytest.approx(min(fids))
    assert result["max_individual_fidelity"] == pytest.approx(max(fids))
    assert result["avg_individual_fidelity"] == pytest.approx(float(np.mean(fids)))


# ---------------------------------------------------------------------------
# Path equivalence: MP and sequential should compute the same per-gate values
#
# Real multiprocessing workers don't inherit monkeypatched module attributes
# (spawn re-imports the module fresh), so we substitute an in-process fake pool
# that runs the worker function inline. This keeps the test deterministic and
# lets the gridsynth mock apply uniformly to both paths.
# ---------------------------------------------------------------------------


class _InProcessPool:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def map(self, fn, args_list):
        return [fn(a) for a in args_list]

    def starmap(self, fn, args_list):
        return [fn(*a) for a in args_list]


def _install_inprocess_pool(monkeypatch):
    monkeypatch.setattr(
        fidelity_mod.multiprocessing, "Pool", lambda *a, **kw: _InProcessPool()
    )


def test_mp_and_sequential_produce_same_individual_fidelities(monkeypatch) -> None:
    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", lambda *_a, **_k: "H")
    _install_inprocess_pool(monkeypatch)
    qc = QuantumCircuit(1)
    qc.rz(0.3, 0)
    qc.rz(0.6, 0)

    seq = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    mp = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=True)

    assert seq["individual_fidelities"] == pytest.approx(mp["individual_fidelities"])
    assert seq["overall_fidelity"] == pytest.approx(mp["overall_fidelity"])
    assert seq["rz_gate_count"] == mp["rz_gate_count"]
    assert mp["multiprocessing_used"] is True


# ---------------------------------------------------------------------------
# failed_decompositions / status accounting — should report identically across
# the multiprocessing, sequential, and SK paths.
# ---------------------------------------------------------------------------


def test_failed_decompositions_counted_in_multiprocessing_path(monkeypatch) -> None:
    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", lambda *_a, **_k: "H")
    _install_inprocess_pool(monkeypatch)
    qc = QuantumCircuit(1)
    qc.rz(0.4, 0)
    qc.rz(0.6, 0)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=True)
    assert result["multiprocessing_used"] is True
    assert result["failed_decompositions"] == 2
    assert result["status"] == "partial_failure"


def test_failed_decompositions_counted_in_sequential_path(monkeypatch) -> None:
    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", lambda *_a, **_k: "H")
    qc = QuantumCircuit(1)
    qc.rz(0.4, 0)
    qc.rz(0.6, 0)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["failed_decompositions"] == 2
    assert result["status"] == "partial_failure"


def test_sk_failed_decompositions_counted(monkeypatch) -> None:
    # Force a low-fidelity SK approximation by replacing the synthesis helper
    def _bad_sk(_theta, _r):
        bad = QuantumCircuit(1)
        bad.h(0)
        return bad

    monkeypatch.setattr(fidelity_mod, "_synthesize_single_rz_with_sk", _bad_sk)
    qc = QuantumCircuit(1)
    qc.rz(0.4, 0)
    qc.rz(0.6, 0)
    result = rz_product_fidelity_sk(qc, recursion_degree=1, use_multiprocessing=False)
    assert result["failed_decompositions"] == 2
    assert result["status"] == "partial_failure"


# ---------------------------------------------------------------------------
# Edge-case angle handling
# ---------------------------------------------------------------------------


def test_identity_decomposition_gives_unit_fidelity(monkeypatch) -> None:
    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", lambda *_a, **_k: "")
    qc = QuantumCircuit(1)
    qc.rz(0.7, 0)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["individual_fidelities"] == [1.0]
    assert result["overall_fidelity"] == pytest.approx(1.0)


def test_w_only_decomposition_gives_unit_fidelity(monkeypatch) -> None:
    # "W" is gridsynth's omega global-phase symbol; should be treated as identity.
    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", lambda *_a, **_k: "WW")
    qc = QuantumCircuit(1)
    qc.rz(0.7, 0)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["individual_fidelities"] == [1.0]


def test_tiny_theta_rz_is_skipped(monkeypatch) -> None:
    # |theta| < 1e-10 should not appear in individual_fidelities and should not
    # invoke gridsynth.
    calls = {"n": 0}

    def _spy(*_a, **_k):
        calls["n"] += 1
        return "T"

    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", _spy)
    qc = QuantumCircuit(1)
    qc.rz(1e-15, 0)
    qc.rz(0.5, 0)
    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert result["rz_gate_count"] == 1
    assert len(result["individual_fidelities"]) == 1
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# SK basic-approximations cache (regression test for module-level lru_cache)
# ---------------------------------------------------------------------------


def test_sk_basic_approximations_cached_across_calls(monkeypatch) -> None:
    fidelity_mod._sk_basic_approximations.cache_clear()
    try:
        sentinel = object()
        monkeypatch.setattr(
            fidelity_mod, "generate_basic_approximations", lambda **_kw: sentinel
        )
        first = fidelity_mod._sk_basic_approximations()
        second = fidelity_mod._sk_basic_approximations()
        assert first is sentinel and second is sentinel
        info = fidelity_mod._sk_basic_approximations.cache_info()
        assert info.hits == 1
        assert info.misses == 1
    finally:
        fidelity_mod._sk_basic_approximations.cache_clear()


# ---------------------------------------------------------------------------
# Decomposer return_decomp_map + dedupe, and fidelity reuse of the map.
#
# The decomposer and fidelity step previously each invoked gridsynth once per
# RZ gate, doing the same work twice. The decomposer now returns a
# {theta_str -> gate_string} map, the gs transpiler stashes it on the
# intermediate circuit's metadata, and rz_product_fidelity reuses it instead
# of re-running the CLI.
# ---------------------------------------------------------------------------


def test_decomposer_returns_decomp_map(monkeypatch) -> None:
    monkeypatch.setattr(decomposer_mod, "_run_gridsynth_cli", lambda *_a, **_k: "T")
    qc = QuantumCircuit(1)
    qc.rz(0.4, 0)
    qc.rz(0.6, 0)

    new_circuit, decomp_map = decompose_rz_gates_gridsynth(
        qc, precision=3, return_decomp_map=True
    )
    assert isinstance(decomp_map, dict)
    assert set(decomp_map) == {"0.4", "0.6"}
    assert all(v == "T" for v in decomp_map.values())


def test_decomposer_dedupes_identical_angles(monkeypatch) -> None:
    calls: list[str] = []

    def _spy(theta_str, *_a, **_k):
        calls.append(theta_str)
        return "T"

    monkeypatch.setattr(decomposer_mod, "_run_gridsynth_cli", _spy)
    qc = QuantumCircuit(1)
    qc.rz(0.5, 0)
    qc.rz(0.5, 0)
    qc.rz(0.5, 0)

    _, decomp_map = decompose_rz_gates_gridsynth(
        qc, precision=3, return_decomp_map=True
    )
    assert len(calls) == 1
    assert decomp_map == {"0.5": "T"}


def test_rz_product_fidelity_uses_explicit_decomp_map_arg(monkeypatch) -> None:
    cli_calls = {"n": 0}

    def _spy(*_a, **_k):
        cli_calls["n"] += 1
        return "H"  # would give low fidelity if hit

    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", _spy)
    qc = QuantumCircuit(1)
    qc.rz(0.4, 0)
    qc.rz(0.6, 0)

    # All angles covered by explicit map -> CLI must not be called.
    decomp_map = {"0.4": "", "0.6": ""}  # empty -> identity branch -> fid 1.0
    result = rz_product_fidelity(
        qc,
        gridsynth_precision=3,
        use_multiprocessing=False,
        decomp_map=decomp_map,
    )
    assert cli_calls["n"] == 0
    assert result["individual_fidelities"] == [1.0, 1.0]


def test_rz_product_fidelity_reads_metadata_decomp_map(monkeypatch) -> None:
    cli_calls = {"n": 0}

    def _spy(*_a, **_k):
        cli_calls["n"] += 1
        return "H"

    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", _spy)
    qc = QuantumCircuit(1)
    qc.rz(0.4, 0)
    qc.rz(0.6, 0)
    qc.metadata = {"gridsynth_decomp": {"0.4": "", "0.6": ""}}

    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    assert cli_calls["n"] == 0
    assert result["individual_fidelities"] == [1.0, 1.0]


def test_rz_product_fidelity_falls_back_to_cli_for_missing_angles(
    monkeypatch,
) -> None:
    cli_calls: list[str] = []

    def _spy(theta_str, *_a, **_k):
        cli_calls.append(theta_str)
        return ""  # identity branch -> fid 1.0

    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", _spy)
    qc = QuantumCircuit(1)
    qc.rz(0.4, 0)
    qc.rz(0.6, 0)

    # Map only covers one of the two angles.
    decomp_map = {"0.4": ""}
    result = rz_product_fidelity(
        qc,
        gridsynth_precision=3,
        use_multiprocessing=False,
        decomp_map=decomp_map,
    )
    assert cli_calls == ["0.6"]
    assert result["individual_fidelities"] == [1.0, 1.0]


# ---------------------------------------------------------------------------
# Closed-form 1-qubit fidelity must agree with qiskit.process_fidelity to
# numerical precision. Locks in that the optimization is purely about speed,
# not behavior.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "theta",
    [0.0, 0.1, 0.3, 0.7, 1.0, np.pi / 8, np.pi / 4, np.pi / 2, np.pi, 2.5, -0.5],
)
@pytest.mark.parametrize("approx_seq", ["", "H", "T", "S", "STH", "HTH", "THTHTHS"])
def test_closed_form_fidelity_matches_process_fidelity(
    theta: float, approx_seq: str
) -> None:
    from qiskit.quantum_info import Operator, process_fidelity

    from ftcircuitbench.decomposer import create_circuit_from_gate_string

    ideal = QuantumCircuit(1)
    ideal.rz(theta, 0)
    ideal_op = Operator(ideal)

    approx_qc = (
        create_circuit_from_gate_string(approx_seq) if approx_seq else QuantumCircuit(1)
    )
    approx_op = Operator(approx_qc)

    expected = float(
        process_fidelity(approx_op, ideal_op, require_cp=False, require_tp=False)
    )
    actual = fidelity_mod._unitary_process_fidelity_1q(approx_op, ideal_op)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_rz_product_fidelity_explicit_arg_overrides_metadata(monkeypatch) -> None:
    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", lambda *_a, **_k: "H")
    qc = QuantumCircuit(1)
    qc.rz(0.5, 0)
    # Metadata says identity (would give fid 1.0); explicit arg says "H" (low).
    qc.metadata = {"gridsynth_decomp": {"0.5": ""}}
    result = rz_product_fidelity(
        qc,
        gridsynth_precision=3,
        use_multiprocessing=False,
        decomp_map={"0.5": "H"},
    )
    # If arg won, fidelity is < 1.0 (H is a bad approximation of RZ(0.5)).
    assert result["individual_fidelities"][0] < 0.5


# ---------------------------------------------------------------------------
# Per-angle fidelity dedupe — repeated angles should share work but the
# per-gate output list must remain length == rz_gate_count.
# ---------------------------------------------------------------------------


def test_rz_product_fidelity_dedupes_repeated_angles(monkeypatch) -> None:
    cli_calls: list[str] = []

    def _spy(theta_str, *_a, **_k):
        cli_calls.append(theta_str)
        return "H"

    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", _spy)
    qc = QuantumCircuit(1)
    for _ in range(5):
        qc.rz(0.5, 0)
    qc.rz(0.7, 0)

    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    # 6 RZ gates but only 2 unique angles: gridsynth invoked twice.
    assert cli_calls == ["0.5", "0.7"]
    # Per-gate output preserves length and order.
    assert len(result["individual_fidelities"]) == 6
    assert result["rz_gate_count"] == 6
    assert result["unique_angle_count"] == 2
    # First five entries are RZ(0.5) -> identical fid; last is RZ(0.7).
    assert (
        result["individual_fidelities"][0]
        == result["individual_fidelities"][1]
        == result["individual_fidelities"][2]
        == result["individual_fidelities"][3]
        == result["individual_fidelities"][4]
    )
    assert result["individual_fidelities"][5] != result["individual_fidelities"][0]


def test_rz_product_fidelity_overall_accounts_for_repeated_angles(
    monkeypatch,
) -> None:
    # Catches the regression where dedupe might collapse the product to
    # one factor per unique angle instead of one per gate occurrence.
    monkeypatch.setattr(fidelity_mod, "_run_gridsynth_cli", lambda *_a, **_k: "H")
    qc = QuantumCircuit(1)
    for _ in range(3):
        qc.rz(0.5, 0)

    result = rz_product_fidelity(qc, gridsynth_precision=3, use_multiprocessing=False)
    per_gate = result["individual_fidelities"][0]
    assert result["overall_fidelity"] == pytest.approx(per_gate**3)


def test_rz_product_fidelity_sk_multiprocessing_fallback_warns(monkeypatch) -> None:
    class _BrokenPool:
        def __enter__(self):
            raise RuntimeError("sk pool setup failed")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        fidelity_mod.multiprocessing,
        "Pool",
        lambda *args, **kwargs: _BrokenPool(),
    )

    def _trivial_sk(_theta, _r):
        return QuantumCircuit(1)  # identity -> fid 1.0

    monkeypatch.setattr(fidelity_mod, "_synthesize_single_rz_with_sk", _trivial_sk)

    qc = QuantumCircuit(2)
    qc.rz(0.2, 0)
    qc.rz(0.3, 1)
    with pytest.warns(RuntimeWarning, match="multiprocessing failed"):
        result = rz_product_fidelity_sk(
            qc, recursion_degree=1, use_multiprocessing=True
        )
    assert result["multiprocessing_used"] is False
    assert result["rz_gate_count"] == 2


# ---------------------------------------------------------------------------
# SK path equivalence (analogous to the gridsynth equivalence test).
# Uses an in-process pool so monkeypatched _synthesize_single_rz_with_sk
# propagates to the "worker".
# ---------------------------------------------------------------------------


def test_sk_mp_and_sequential_produce_same_individual_fidelities(
    monkeypatch,
) -> None:
    def _stub_sk(_theta, _r):
        bad = QuantumCircuit(1)
        bad.h(0)
        return bad

    monkeypatch.setattr(fidelity_mod, "_synthesize_single_rz_with_sk", _stub_sk)
    _install_inprocess_pool(monkeypatch)
    qc = QuantumCircuit(1)
    qc.rz(0.3, 0)
    qc.rz(0.6, 0)

    seq = rz_product_fidelity_sk(qc, recursion_degree=1, use_multiprocessing=False)
    mp = rz_product_fidelity_sk(qc, recursion_degree=1, use_multiprocessing=True)
    assert seq["individual_fidelities"] == pytest.approx(mp["individual_fidelities"])
    assert seq["overall_fidelity"] == pytest.approx(mp["overall_fidelity"])
    assert mp["multiprocessing_used"] is True


# ---------------------------------------------------------------------------
# Boundary at MAX_QUBITS_FOR_FIDELITY: small -> unitary path, large -> rz product
# ---------------------------------------------------------------------------


def test_calculate_circuit_fidelity_dispatches_at_qubit_boundary() -> None:
    from ftcircuitbench.fidelity import MAX_QUBITS_FOR_FIDELITY

    small = QuantumCircuit(MAX_QUBITS_FOR_FIDELITY)
    small.h(0)
    small_result = calculate_circuit_fidelity(
        small, small.copy(), gridsynth_precision=3
    )
    assert small_result["method"] == "unitary_based"
    assert small_result["fidelity"] == pytest.approx(1.0, abs=1e-12)

    # n+1 qubits with no intermediate -> falls through to "N/A" via product path
    large = QuantumCircuit(MAX_QUBITS_FOR_FIDELITY + 1)
    large.rz(0.3, 0)
    large_result = calculate_circuit_fidelity(
        large, large.copy(), gridsynth_precision=3, intermediate_qc=None
    )
    assert large_result["method"] in {
        "rz_product_fidelity",
        "rz_product_fidelity_sk",
    }
    assert large_result["fidelity"] == "N/A"


def test_rz_product_fidelity_sk_dedupes_repeated_angles(monkeypatch) -> None:
    sk_calls: list[float] = []

    def _spy(theta_value, _r):
        sk_calls.append(theta_value)
        bad = QuantumCircuit(1)
        bad.h(0)
        return bad

    monkeypatch.setattr(fidelity_mod, "_synthesize_single_rz_with_sk", _spy)
    qc = QuantumCircuit(1)
    for _ in range(4):
        qc.rz(0.5, 0)
    qc.rz(0.9, 0)

    result = rz_product_fidelity_sk(qc, recursion_degree=1, use_multiprocessing=False)
    # 5 RZ gates, 2 unique angles -> SK synthesis runs twice.
    assert sk_calls == [0.5, 0.9]
    assert len(result["individual_fidelities"]) == 5
    assert result["rz_gate_count"] == 5
    assert result["unique_angle_count"] == 2
    # Overall fidelity reflects per-occurrence multiplication.
    fid_5 = result["individual_fidelities"][0]
    fid_9 = result["individual_fidelities"][4]
    assert result["overall_fidelity"] == pytest.approx(fid_5**4 * fid_9)


# ---------------------------------------------------------------------------
# Slow integration tests against real Solovay-Kitaev / real gridsynth.
# Run with `pytest -m slow`. Skipped by default in CI unless explicitly opted in.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_sk_synthesis_produces_high_fidelity_circuit() -> None:
    fidelity_mod._sk_basic_approximations.cache_clear()
    try:
        approx_qc = fidelity_mod._synthesize_single_rz_with_sk(
            theta_value=0.5, recursion_degree=1
        )
        assert isinstance(approx_qc, QuantumCircuit)
        assert approx_qc.num_qubits == 1
        # Synthesis should produce *some* gates for a non-trivial rotation
        assert len(approx_qc.data) > 0

        ideal = QuantumCircuit(1)
        ideal.rz(0.5, 0)
        from qiskit.quantum_info import Operator

        fid = fidelity_mod._unitary_process_fidelity_1q(
            Operator(approx_qc), Operator(ideal)
        )
        # SK at recursion_degree=1 is rough but should beat random (~0.5)
        assert fid > 0.7
    finally:
        fidelity_mod._sk_basic_approximations.cache_clear()


@pytest.mark.slow
def test_real_gridsynth_end_to_end() -> None:
    import shutil

    if shutil.which("gridsynth") is None:
        pytest.skip("gridsynth not installed")
    qc = QuantumCircuit(1)
    qc.rz(0.5, 0)
    result = rz_product_fidelity(qc, gridsynth_precision=5, use_multiprocessing=False)
    assert result["rz_gate_count"] == 1
    assert isinstance(result["overall_fidelity"], float)
    # gridsynth at precision=5 should give a very good approximation
    assert result["overall_fidelity"] > 0.99
