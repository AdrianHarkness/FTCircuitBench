"""Microbenchmark for the PBC layering algorithm.

Two layers of measurement:

  1. Microbench  — times `TableauPauliBasis.layer*` directly on synthetic
                   tableaus of varying size and Pauli weight. This is the loop
                   we are actually optimizing.
  2. Macrobench  — times the full `RotationPauliCirc.optimize_t` loop on
                   seeded random Clifford+T circuits, so we can confirm that
                   any microbench win actually translates to a pipeline win.

Run from repo root:

    .venv/bin/python -m bench.bench_layering              # human-readable table
    .venv/bin/python -m bench.bench_layering --json out.json  # also dump JSON

To establish a baseline before a refactor, save the JSON. After the refactor,
re-run with the same seed and diff. Lower is better.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, List

import numpy as np
from qiskit import QuantumCircuit

# Silence tqdm so the inner progress bars don't add noise / pollute output.
import ftcircuitbench.pbc_converter.r_pauli_circ as _rpc_mod
import ftcircuitbench.pbc_converter.tab_gate as _tab_gate_mod

_rpc_mod.tqdm = lambda iterable, **_kwargs: iterable
_tab_gate_mod.tqdm = lambda iterable, **_kwargs: iterable

from ftcircuitbench.pbc_converter.r_pauli_circ import RotationPauliCirc  # noqa: E402
from ftcircuitbench.pbc_converter.tab_gate import TableauPauliBasis  # noqa: E402


# ---------- Inputs ----------


def random_pauli_tableau(
    num_qubits: int, num_rotations: int, weight: int, seed: int
) -> TableauPauliBasis:
    """Build a random TableauPauliBasis with controlled Pauli weight.

    `weight` is the number of non-identity positions per row. Each non-I position
    is independently X, Y, or Z with equal probability.
    """
    rng = np.random.default_rng(seed)
    rows = np.zeros((num_rotations, 2 * num_qubits + 1), dtype=bool)
    for r in range(num_rotations):
        positions = rng.choice(num_qubits, size=weight, replace=False)
        for q in positions:
            choice = rng.integers(3)  # 0=X, 1=Z, 2=Y
            if choice == 0:
                rows[r, q] = True
            elif choice == 1:
                rows[r, num_qubits + q] = True
            else:
                rows[r, q] = True
                rows[r, num_qubits + q] = True
        rows[r, -1] = bool(rng.integers(2))
    return TableauPauliBasis(rows)


def random_clifford_t_circuit(
    num_qubits: int, num_ops: int, t_fraction: float, seed: int
) -> QuantumCircuit:
    """Generate a deterministic Clifford+T circuit."""
    rng = random.Random(seed)
    qc = QuantumCircuit(num_qubits)
    t_gates = ["t", "tdg"]
    cliff_1q = ["h", "s"]
    for _ in range(num_ops):
        roll = rng.random()
        if roll < 0.2 and num_qubits >= 2:
            a, b = rng.sample(range(num_qubits), 2)
            qc.cx(a, b)
        elif roll < 0.2 + t_fraction:
            getattr(qc, rng.choice(t_gates))(rng.randrange(num_qubits))
        else:
            getattr(qc, rng.choice(cliff_1q))(rng.randrange(num_qubits))
    return qc


# ---------- Timing ----------


def best_of(fn: Callable[[], object], runs: int) -> float:
    """Run `fn` `runs` times, return best wall-clock seconds (ignores result)."""
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    return min(timings)


# ---------- Cases ----------


@dataclass
class Result:
    case: str
    method: str
    best_seconds: float
    rotations: int
    layers: int
    extra: dict


def _layer_case(
    case_name: str, tab_factory: Callable[[], TableauPauliBasis], runs: int
) -> List[Result]:
    """Run the four layering methods on a fresh tableau each time."""
    out: List[Result] = []

    # Snapshot one run for layer-count reporting (use a fresh tab each method).
    def _run_method(method: str) -> tuple[float, int]:
        # Each run gets its own tableau because layering mutates internal lists.
        layers_holder: list = []

        def _do() -> None:
            tab = tab_factory()
            if method == "bare":
                layers_holder[:] = [tab.layer()]
            elif method == "v2":
                layers_holder[:] = [tab.layer_v2()]
            elif method == "v2_max16":
                layers_holder[:] = [tab.layer_v2(max_layer_checks=16)]
            elif method == "v2_max4":
                layers_holder[:] = [tab.layer_v2(max_layer_checks=4)]

        best = best_of(_do, runs)
        return best, len(layers_holder[0])

    sample_tab = tab_factory()
    rotations = sample_tab.stab_counts

    for method in ["bare", "v2", "v2_max16", "v2_max4"]:
        best, layers = _run_method(method)
        out.append(
            Result(
                case=case_name,
                method=method,
                best_seconds=best,
                rotations=rotations,
                layers=layers,
                extra={},
            )
        )
    return out


def _optimize_t_case(
    case_name: str, qc: QuantumCircuit, runs: int
) -> List[Result]:
    """Time the full optimize_t loop for each layering method."""
    out: List[Result] = []
    for method in ["bare", "v2"]:
        tracker_holder: list = []

        def _do() -> None:
            rpc = RotationPauliCirc(qc)
            err = rpc.process(ifprint=False)
            assert err is False
            tracker_holder[:] = [
                rpc.optimize_t(maxiter=10, ifprint=False, layering_method=method)
            ]

        best = best_of(_do, runs)
        tracker = tracker_holder[0]
        out.append(
            Result(
                case=case_name,
                method=f"optimize_t/{method}",
                best_seconds=best,
                rotations=tracker[0],
                layers=tracker[-1],
                extra={"trace": tracker},
            )
        )
    return out


# ---------- Driver ----------


SEED = 20260508


def build_cases(runs: int) -> List[Result]:
    results: List[Result] = []

    # ----- Microbench: synthetic tableaus, varied size and weight -----
    micro_specs = [
        # (label, qubits, rotations, weight)
        ("micro/small_sparse",  10,  200, 2),
        ("micro/small_dense",   10,  200, 8),
        ("micro/medium_sparse", 20, 1000, 2),
        ("micro/medium_dense",  20, 1000, 12),
        ("micro/large_sparse",  40, 3000, 3),
        ("micro/large_dense",   40, 3000, 20),
    ]
    for label, q, n, w in micro_specs:
        factory = lambda q=q, n=n, w=w: random_pauli_tableau(q, n, w, SEED)
        results.extend(_layer_case(label, factory, runs))

    # ----- Macrobench: full optimize_t on real Clifford+T circuits -----
    macro_specs = [
        # (label, qubits, ops, t_fraction)
        ("macro/small",  6,  200,  0.40),
        ("macro/medium", 12, 1000, 0.40),
        ("macro/large",  20, 3000, 0.40),
    ]
    for label, q, ops, frac in macro_specs:
        qc = random_clifford_t_circuit(q, ops, frac, SEED)
        results.extend(_optimize_t_case(label, qc, runs=max(1, runs // 2)))

    return results


def print_table(results: List[Result]) -> None:
    print(f"{'case':<22} {'method':<22} {'best (ms)':>10} "
          f"{'rotations':>10} {'layers':>8}")
    print("-" * 76)
    for r in results:
        print(f"{r.case:<22} {r.method:<22} "
              f"{r.best_seconds*1000:>10.2f} {r.rotations:>10} {r.layers:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3,
                        help="repetitions per case; reports the best (default 3)")
    parser.add_argument("--json", type=str, default=None,
                        help="optional path to dump full results as JSON")
    args = parser.parse_args()

    print(f"Running layering benchmark (seed={SEED}, runs={args.runs})")
    print()

    results = build_cases(args.runs)
    print_table(results)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {"seed": SEED, "runs": args.runs,
                 "results": [asdict(r) for r in results]},
                f, indent=2,
            )
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
