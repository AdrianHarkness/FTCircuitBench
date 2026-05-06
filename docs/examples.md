# Examples

Three worked examples, in increasing levels of detail:

1. [Single-circuit CLI run](#1-single-circuit-cli-analyze_circuitpy) via `analyze_circuit.py`.
2. [Batch benchmark CLI run](#2-batch-benchmarks-generate_benchmarkspy) via `generate_benchmarks.py`.
3. [Programmatic usage](#3-python-api-via-ftcircuitbenchapi) of `ftcircuitbench.api`.

All three use the same 4-qubit QFT smoke-test circuit (`qasm/qft/qft_4q.qasm`)
that the README's reproducibility section uses, so results are directly
comparable.

---

## 1. Single-circuit CLI (`analyze_circuit.py`)

Run the Gridsynth + PBC pipeline on a single circuit:

```bash
uv run python analyze_circuit.py qasm/qft/qft_4q.qasm \
  --pipeline gs \
  --gridsynth-precision 5 \
  --skip-fidelity
```

Outputs are written under three top-level directories (paths derived from the
input filename and pipeline parameters):

- `circuit_stats_output/qft_4q_gs_prec5_stats.json` — aggregated Clifford+T and PBC statistics.
- `clifford_t_output/qft_4q_gs_prec5_clifford_t.qasm` — the transpiled Clifford+T circuit.
- `pbc_output/qft_4q_gs_prec5_*.txt` — PBC measurement bases and T-layers.

A successful run takes a few seconds on a recent laptop. Common flags:
`--pipeline {gs,sk,both}`, `--sk-recursion N`, `--layering-max-checks K`,
`--optimize-pbc`, `--optimize-t-maxiter N`, `--max-workers N`. See
`uv run python analyze_circuit.py --help` for the full list.

---

## 2. Batch benchmarks (`generate_benchmarks.py`)

Reproduce the full benchmark sweep across every circuit in `qasm/`:

```bash
uv run python generate_benchmarks.py
```

Approximate runtime: several hours on a workstation; longer on a laptop. The
output structure mirrors the committed reference layout under
`circuit_benchmarks/`. See `uv run python generate_benchmarks.py --help` for
the full set of flags (for example to restrict the sweep to a subset of
circuits or to a particular pipeline).

---

## 3. Python API via `ftcircuitbench.api`

The same pipeline can be driven programmatically. The snippet below was
verified by running it under `uv run python` against the current `main`
branch.

```python
from ftcircuitbench.api import PipelineConfig, run_analysis_for_file

config = PipelineConfig(
    pipeline="gs",
    gridsynth_precision=3,
    calculate_fidelity=False,
)

analysis = run_analysis_for_file("qasm/qft/qft_4q.qasm", config)
gs = analysis.pipelines["gs"]

print("original_qubits:", analysis.original_qubits)
print("original_gates:", analysis.original_gates)
print("t_count:", gs.clifford_stats["t_count"])
print("total_t_family_count:", gs.clifford_stats["total_t_family_count"])
```

Expected output (gate counts depend on the active `nwqec` / `gridsynth`
backend; the structure should match):

```
original_qubits: 4
original_gates: 17
t_count: 264
total_t_family_count: 267
```

To run multiple pipelines on the same circuit, pass a list of `PipelineConfig`
objects:

```python
configs = [
    PipelineConfig(pipeline="gs", gridsynth_precision=4, optimize_pbc=True),
    PipelineConfig(pipeline="sk", sk_recursion=2, calculate_fidelity=False),
]
analysis = run_analysis_for_file("qasm/qft/qft_4q.qasm", configs)
print(analysis.pipelines["gs"].pbc_stats.get("pbc_rotation_operators"))
print(analysis.to_dict(include_artifacts=True))
```

Each `PipelineResult` exposes `clifford_t_circuit`, `pbc_circuit`,
`clifford_stats`, `pbc_stats`, optional `fidelity`, per-stage `timings`, and a
`to_dict()` accessor. See [`api.md`](api.md) for the full reference and
[`installation.md`](installation.md) for setup.

For an annotated, cell-by-cell walkthrough see
`FTCircuitBench_Pipeline_Demo.ipynb` in the repository root.
