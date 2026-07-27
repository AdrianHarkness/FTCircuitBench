# Examples

Four worked examples, in increasing levels of detail:

1. [Single-circuit CLI run](#1-single-circuit-cli-analyze_circuitpy) via `analyze_circuit.py`.
2. [Batch benchmark CLI run](#2-batch-benchmarks-generate_benchmarkspy) via `generate_benchmarks.py`.
3. [Programmatic usage](#3-python-api-via-ftcircuitbenchapi) of `ftcircuitbench.api`.
4. [Physical resource estimation](#4-physical-resource-estimation-estimate_resourcespy) via `estimate_resources.py`.

The first three use the same 4-qubit QFT smoke-test circuit
(`qasm/qft/qft_4q.qasm`) that the README's reproducibility section uses, so
results are directly comparable.

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

---

## 4. Physical resource estimation (`estimate_resources.py`)

The three examples above stop at the logical layer. To ask what a circuit costs
*after* error correction — physical qubits, wall-clock runtime, code distance,
T factories — hand the logical counts to the Azure Quantum Resource Estimator.
This needs the optional `qre` extra:

```bash
uv sync --extra qre       # or: pip install "ftcircuitbench[qre]"
```

The estimator is bundled in the `qdk` package and runs locally; there is no
Azure subscription or network call involved.

### CLI

```bash
# every circuit in the aggregated benchmark CSV, two superconducting models
uv run python estimate_resources.py

# a subset, by glob on the benchmark label
uv run python estimate_resources.py --label 'qft-*' --label 'adder-64q-*'

# one circuit straight from analyze_circuit.py output, priced as PBC
uv run python estimate_resources.py \
  --stats-json circuit_stats_output/qft_4q_gs_prec5_stats.json \
  --counts pbc --model 'majorana 1e-6'
```

Results are written to `qre_output/qre_results.json` (override with `--output`)
as one record per circuit:

```json
{
  "label": "qft-29q-sk-2",
  "num_qubits": 29,
  "t_count": 7872,
  "measurement_count": 29,
  "counts_source": "clifford_t",
  "models": {
    "superconducting 1e-3": {
      "physical_qubits": 131830,
      "runtime_s": 0.0410852,
      "code_distance": 13,
      "logical_depth": 7901,
      "algorithmic_logical_qubits": 75,
      "num_t_states": 7872,
      "num_t_factories": 11,
      "physical_qubits_for_algorithm": 25350,
      "physical_qubits_for_t_factories": 106480
    }
  }
}
```

Useful flags: `--list-models` (the six built-in hardware models),
`--model NAME` (repeatable), `--error-budget F`, `--summary-model NAME`,
`--quiet`. Clifford-only circuits are skipped with a warning — Azure QRE has no
T factory to lay out when the T count is zero.

### Python API

```python
from ftcircuitbench.resource_estimation import (
    estimate_circuits,
    read_ct_stats_csv,
    resolve_hardware_models,
)

counts = read_ct_stats_csv("circuit_benchmarks/ct_stats.csv")
models = resolve_hardware_models(["superconducting 1e-4"], error_budget=0.001)

for result in estimate_circuits(counts[:5], models=models):
    estimate = result.estimates["superconducting 1e-4"]
    print(
        f"{result.label:24s} "
        f"{estimate.physical_qubits:>10,} physical qubits  "
        f"d={estimate.code_distance}  "
        f"{estimate.runtime_seconds:.3f}s"
    )
```

To go straight from a pipeline run to an estimate, build the counts from the
stats a pipeline produced:

```python
from ftcircuitbench.api import PipelineConfig, run_analysis_for_file
from ftcircuitbench.resource_estimation import estimate_circuit, logical_counts_from_stats

analysis = run_analysis_for_file(
    "qasm/qft/qft_4q.qasm",
    PipelineConfig(pipeline="gs", gridsynth_precision=5, calculate_fidelity=False),
)
gs = analysis.pipelines["gs"]
stats = {**gs.clifford_stats, **gs.pbc_stats}

counts = logical_counts_from_stats(stats, label="qft-4q-gs-5")
print(estimate_circuit(counts).to_dict())
```

Pass `counts_source="pbc"` to price the post-optimization PBC rotation and
measurement operators instead of the Clifford+T T-family count.

Two caveats worth keeping in mind when reading the numbers:

- FTCircuitBench has already synthesised every `rz` into Clifford+T, so the
  counts carry a concrete `tCount` and leave `rotationCount` at zero. The
  estimate reflects FTCircuitBench's synthesis at the precision you chose, not
  QRE's internal rotation-cost model.
- Unless a stats file says otherwise, `measurement_count` is taken to be the
  circuit width — one terminal measurement per qubit.

---

For an annotated, cell-by-cell walkthrough see
`FTCircuitBench_Pipeline_Demo.ipynb` in the repository root.
