# Examples

Five worked examples, in increasing levels of detail:

1. [Single-circuit CLI run](#1-single-circuit-cli-analyze_circuitpy) via `analyze_circuit.py`.
2. [Batch benchmark CLI run](#2-batch-benchmarks-generate_benchmarkspy) via `generate_benchmarks.py`.
3. [Programmatic usage](#3-python-api-via-ftcircuitbenchapi) of `ftcircuitbench.api`.
4. [Physical resource estimation](#4-physical-resource-estimation-estimate_resourcespy) via `estimate_resources.py`.
5. [Importing circuits from Qualtran and pyLIQTR](#5-importing-circuits-from-qualtran-and-pyliqtr) via `ftcircuitbench.frontends`.

The first three use the same 4-qubit QFT smoke-test circuit
(`qasm/qft/qft_4q.qasm`) that the README's reproducibility section uses, so
results are directly comparable. Examples 4 and 5 are the two ends of the stack:
where circuits come from, and what they cost on hardware.

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

## 5. Importing circuits from Qualtran and pyLIQTR

The examples above start from QASM already in `qasm/`. Fault-tolerant algorithms
are usually *constructed* somewhere else — Qualtran and pyLIQTR both build them
and emit Cirq circuits. `ftcircuitbench.frontends` brings those in through
OpenQASM 2.

```bash
uv sync --extra qualtran      # pulls cirq-core; or: uv sync --all-extras
```

### CLI

```bash
# a Qualtran Bloq, exported and analysed in one step
uv run python import_circuit.py qualtran.bloqs.qft:QFTTextBook --args 4 \
  --unitary-uncompute --output qasm/imported/qft_4.qasm --analyze

# `--args` / `--kwargs` are JSON, so simple constructors work directly:
uv run python import_circuit.py qualtran.bloqs.mcmt:And --kwargs '{"uncompute": false}' \
  -o qasm/imported/and.qasm
```

JSON has no tuple, and many Qualtran constructors require one. For anything
beyond scalar arguments, point the CLI at a factory function instead — the
target may be any dotted path, and any callable returning a Bloq or a Cirq
circuit:

```python
# my_bloqs.py
import numpy as np
from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM

DATA = np.random.default_rng(2026).integers(0, 32, size=32)

def qroam(log_block_size: int = 1):
    return SelectSwapQROM.build_from_data(
        DATA, target_bitsizes=(5,), log_block_sizes=(log_block_size,)
    )
```

```bash
uv run python import_circuit.py my_bloqs:qroam --args 2 \
  --unitary-uncompute -o qasm/imported/qroam.qasm
```

### Python API

```python
from ftcircuitbench.api import PipelineConfig, run_pipeline
from ftcircuitbench.frontends import bloq_t_complexity, bloq_to_qiskit
from qualtran.bloqs.qft import QFTTextBook

bloq = QFTTextBook(4)
circuit = bloq_to_qiskit(bloq, unitary_uncompute=True)
result = run_pipeline(
    circuit, PipelineConfig(pipeline="gs", gridsynth_precision=5,
                            calculate_fidelity=False)
)

print("qualtran analytic:", bloq_t_complexity(bloq))
print("ftcircuitbench compiled T:", result.clifford_stats["total_t_family_count"])
```

A plain Cirq circuit goes through `cirq_to_qiskit` instead:

```python
import cirq
from ftcircuitbench.frontends import cirq_to_qiskit

q = cirq.LineQubit.range(3)
circuit = cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.T(q[2])])
qc = cirq_to_qiskit(circuit)
```

### Two behaviours worth knowing

**Decomposition stops at OpenQASM 2.** `cirq.decompose` with no stopping rule
lowers to a hardware gateset, rewriting `H` and `CNOT` into `X`/`Y`/`Z` power
gates and inflating the circuit several-fold (a 4-qubit QFT becomes 1943 gates
at depth 946 instead of 519 at depth 370). The frontend stops as soon as every
operation has a QASM representation, so the exported circuit stays compact and
close to what the generator emitted. Compiled resource counts are the same
either way — the extra rotations a full decomposition produces carry Clifford+T
angles that synthesis reproduces exactly — so the stopping rule is about the
size and fidelity of the intermediate circuit, not the final T count.

**Non-unitary circuits are refused, not guessed at.** Qualtran's `And` adjoint is
implemented by measurement and classical fix-up — that is why it costs zero T
gates, and why the decomposed circuit is not unitary. FTCircuitBench's Clifford+T
and PBC analysis models unitary circuits, so:

```python
bloq_to_qiskit(bloq)                          # NonUnitaryCircuitError
bloq_to_qiskit(bloq, unitary_uncompute=True)  # substitutes the unitary And†
```

The substitution is exact — it is the `And` compute circuit reversed and
inverted. (Qualtran's `And` is Gidney's temporary AND, whose target starts in
|0⟩; that assumption is what makes its unitary adjoint cost 4 T rather than the
7 T of a general ancilla-free Toffoli.) Its price closes exactly:

```python
from ftcircuitbench.frontends import bloq_t_complexity, count_measurement_uncompute

analytic = bloq_t_complexity(bloq)["t"]
substitutions = count_measurement_uncompute(bloq)
measured = sum(bloq_to_qiskit(bloq, unitary_uncompute=True).count_ops()[g]
               for g in ("t", "tdg"))
assert measured == analytic + 4 * substitutions
```

Numbers produced this way are an upper bound on the measurement-based
implementation, by exactly `4 x substitutions` T gates.

### pyLIQTR

pyLIQTR emits Cirq circuits too, but its releases pin `numpy<2` and
`qualtran==0.4.0`, which cannot coexist with FTCircuitBench's dependency floor.
The circuit crosses as a file. `tools/export_cirq_qasm.py` imports nothing from
`ftcircuitbench` and applies the same decomposition and unitarity rules, so the
QASM matches what the in-process path would produce:

```bash
# in the pyLIQTR environment
python tools/export_cirq_qasm.py my_module:build_encoding \
  --unitary-uncompute -o heisenberg_be.qasm

# in the FTCircuitBench environment
uv run python analyze_circuit.py heisenberg_be.qasm --pipeline gs
uv run python estimate_resources.py \
  --stats-json circuit_stats_output/heisenberg_be_gs_prec5_stats.json
```

`my_module:build_encoding` names any callable returning a Cirq circuit, gate, or
operation — for example a `getEncoding(VALID_ENCODINGS.PauliLCU)(model)` applied
to its qubits.

---

For an annotated, cell-by-cell walkthrough see
`demos/FTCircuitBench_Pipeline_Demo.ipynb`, and
`demos/Qualtran_to_QRE_Demo.ipynb` for the full Qualtran → FTCircuitBench →
Azure QRE path on a Qualtran QFT.
