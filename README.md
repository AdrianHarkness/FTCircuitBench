# FTCircuitBench

[![CI](https://github.com/pnnl/FTCircuitBench/actions/workflows/ci.yml/badge.svg)](https://github.com/pnnl/FTCircuitBench/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2601.03185-b31b1b.svg)](https://arxiv.org/abs/2601.03185)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A benchmark suite for fault-tolerant quantum circuit compilation and architecture, covering Clifford+T synthesis (Gridsynth and Solovay-Kitaev) and Pauli-Based Computation (PBC).

FTCircuitBench sits between the tools that *construct* logical circuits and the tools that *price* physical ones:

```
Qualtran / pyLIQTR  ──▶  FTCircuitBench  ──▶  Azure QRE
  (logical circuits)      Clifford+T, PBC,     (physical qubits,
                          structure, stats      runtime, code distance)
```

See [`demos/Qualtran_to_QRE_Demo.ipynb`](demos/Qualtran_to_QRE_Demo.ipynb) for that path end to end on a Qualtran QFT.

## Install

```bash
git clone https://github.com/pnnl/FTCircuitBench.git
cd FTCircuitBench
uv sync --all-extras       # creates .venv with all deps + dev tools
uv run pytest              # verify install
```

This uses [uv](https://docs.astral.sh/uv/) and the committed `uv.lock` for a
reproducible install. The pinned interpreter is read from `.python-version`
(currently `3.11`).

### pip alternative

If you prefer pip:

```bash
pip install -e ".[dev]"
```

Requirements: Python 3.10+, [`nwqec`](https://github.com/pnnl/nwqec) (for fast Gridsynth/PBC via `fuse_t`). An optional `gridsynth` binary on your `PATH` enables the Python-fallback GS path.

Optional extras, all included by `uv sync --all-extras`:

| Extra | Installs | Enables |
|---|---|---|
| `qre` | [`qdk`](https://pypi.org/project/qdk/) | [Physical resource estimation](#physical-resource-estimation-azure-qre) via the Azure Quantum Resource Estimator |
| `cirq` | [`cirq-core`](https://quantumai.google/cirq) | [Ingesting circuits](#logical-circuit-frontends-qualtran-pyliqtr) from any Cirq-based generator |
| `qualtran` | [`qualtran`](https://github.com/quantumlib/Qualtran) | Lowering Qualtran Bloqs straight into the pipeline |

## Quick start

Analyze one circuit (Gridsynth pipeline, PBC on):

```bash
uv run python analyze_circuit.py qasm/qft/qft_18q.qasm \
  --pipeline gs \
  --gridsynth-precision 5
```

Run the full benchmark suite:

```bash
uv run python generate_benchmarks.py
```

Estimate physical (post-error-correction) resources for the benchmark suite:

```bash
uv run python estimate_resources.py --label 'qft-29q-*'
```

Open a notebook:

```bash
jupyter notebook demos/FTCircuitBench_Pipeline_Demo.ipynb   # pipeline walkthrough
jupyter notebook demos/Qualtran_to_QRE_Demo.ipynb           # Qualtran -> FTCB -> Azure QRE
```

Select the project `.venv` kernel and run all cells.

**Common CLI flags:** `--pipeline {gs,sk,both}`, `--gridsynth-precision N`, `--sk-recursion N`, `--layering-max-checks K`, `--optimize-pbc`, `--optimize-t-maxiter N`, `--skip-fidelity`, `--max-workers N`.

## Reproducing paper results

FTCircuitBench accompanies the paper at [arXiv:2601.03185](https://arxiv.org/abs/2601.03185). The benchmark inputs in `qasm/` and reference outputs in `circuit_outputs/` were used to produce the figures and tables in the paper.

### Smoke-test reproduction (one circuit)

To verify your install reproduces a known result, run the Gridsynth + Clifford+T pipeline on the smallest QFT circuit:

```bash
# Run the Gridsynth + Clifford+T pipeline on a 4-qubit QFT
uv run python analyze_circuit.py qasm/qft/qft_4q.qasm \
  --pipeline gs \
  --gridsynth-precision 5 \
  --skip-fidelity
```

This writes three artifacts to the working tree (paths derived from the input filename and pipeline parameters):

- `circuit_stats_output/qft_4q_gs_prec5_stats.json` — aggregated Clifford+T and PBC statistics
- `clifford_t_output/qft_4q_gs_prec5_clifford_t.qasm` — the transpiled Clifford+T circuit
- `pbc_output/qft_4q_gs_prec5_*.txt` — PBC measurement bases and T-layers

A successful run takes a few seconds on a recent laptop and the JSON should begin with:

```json
{
  "t_count": 456,
  "tdg_count": 3,
  "total_t_family_count": 459,
  "compilation_precision_digits": 5,
  ...
}
```

### Full benchmark reproduction

To reproduce the full set of benchmarks behind `circuit_benchmarks/` and the figures in the paper:

```bash
uv run python generate_benchmarks.py
```

This iterates over all circuits in `qasm/` and produces aggregated statistics. **Approximate runtime: several hours on a workstation; longer on a laptop.** The output reproduces the structure under `circuit_benchmarks/`.

See [`docs/examples.md`](docs/examples.md) for additional usage patterns, including how to run a subset of circuits or sweep over compilation parameters.

## Python API

```python
from ftcircuitbench.api import PipelineConfig, run_analysis_for_file

cfgs = [
    PipelineConfig(pipeline="gs", gridsynth_precision=5, optimize_pbc=True),
    PipelineConfig(pipeline="sk", sk_recursion=2),
]
result = run_analysis_for_file("qasm/qft/qft_18q.qasm", cfgs)
print(result.pipelines["gs"].clifford_stats["total_t_family_count"])
```

See [`docs/api.md`](docs/api.md) for the full API reference.

## Logical-circuit frontends (Qualtran, pyLIQTR)

Qualtran and pyLIQTR construct fault-tolerant algorithms and emit Cirq circuits.
`ftcircuitbench.frontends` lowers those to OpenQASM 2 and into the pipeline, so a
Bloq can be compiled, converted to PBC, and costed at the physical layer without
leaving Python:

```bash
uv sync --extra qualtran            # pulls cirq-core too

uv run python import_circuit.py qualtran.bloqs.qft:QFTTextBook --args 4 \
  --unitary-uncompute --output qasm/imported/qft_4.qasm --analyze
```

```python
from ftcircuitbench.api import PipelineConfig, run_pipeline
from ftcircuitbench.frontends import bloq_to_qiskit
from qualtran.bloqs.qft import QFTTextBook

circuit = bloq_to_qiskit(QFTTextBook(4), unitary_uncompute=True)
result = run_pipeline(circuit, PipelineConfig(pipeline="gs", gridsynth_precision=5))
print(result.clifford_stats["total_t_family_count"])
```

Two things the frontend does deliberately:

- **It stops decomposing as soon as OpenQASM 2 can express every operation.**
  Running `cirq.decompose` to the bottom rewrites `H`/`CNOT` into rotation
  gates and inflates the circuit several-fold. Stopping early keeps the
  exported circuit compact and close to the Clifford+T structure the generator
  emitted. (Compiled resource counts are the same either way — the extra
  rotations carry Clifford+T angles that synthesis reproduces exactly.)
- **It refuses non-unitary circuits by default.** Qualtran's `And` adjoint is
  measurement-based, which is why it costs zero T gates and why the decomposed
  circuit is not a unitary. `unitary_uncompute=True` substitutes the unitary
  adjoint instead, at a price that closes exactly: 4 T gates per substitution
  (Qualtran's `And` is Gidney's temporary AND with a |0⟩-initialised target,
  which is what makes its unitary adjoint 4 T rather than a general Toffoli's
  7), so `measured T == qualtran analytic T + 4 x substitutions`.

**pyLIQTR** emits Cirq circuits too, but its releases pin `numpy<2` and
`qualtran==0.4.0`, which cannot coexist with FTCircuitBench's dependency floor.
Cross the boundary as a file instead — `tools/export_cirq_qasm.py` imports
nothing from FTCircuitBench and applies the same rules:

```bash
python tools/export_cirq_qasm.py my_module:build_encoding \
  --unitary-uncompute -o heisenberg_be.qasm        # pyLIQTR environment
uv run python analyze_circuit.py heisenberg_be.qasm --pipeline gs   # here
```

## Physical resource estimation (Azure QRE)

FTCircuitBench reports *logical* costs: Clifford+T gate counts and PBC
rotation/measurement operators. `estimate_resources.py` carries those counts
down to the *physical* layer — physical qubits, wall-clock runtime, surface-code
distance, and T-factory count — via the [Azure Quantum Resource
Estimator](https://learn.microsoft.com/azure/quantum/intro-to-resource-estimation)
that ships inside the `qdk` package. The estimator runs locally; no Azure
subscription or network access is needed.

```bash
uv sync --extra qre                                     # one-time install

uv run python estimate_resources.py --list-models
uv run python estimate_resources.py --label 'heisenberg-1d-100q-*'
```

By default this reads `circuit_benchmarks/ct_stats.csv` (written by
`generate_benchmarks.py`) and estimates every circuit under the two
superconducting models, writing `qre_output/qre_results.json`:

```
=== Physical Resources (superconducting 1e-3) ===
Circuit                    T count  Physical qubits  Runtime  Code distance
-----------------------  ---------  ---------------  -------  -------------
heisenberg-1d-100q-gs-5    575,080          384,940    3.91s             17
heisenberg-1d-100q-gs-8    966,360          400,060    7.35s             19
heisenberg-1d-100q-sk-1    399,840          384,940    2.72s             17
heisenberg-1d-100q-sk-2    528,160          384,940    3.59s             17
```

To estimate a single circuit straight from `analyze_circuit.py` output — and to
price the PBC execution model rather than the Clifford+T one — point it at a
stats JSON:

```bash
uv run python estimate_resources.py \
  --stats-json circuit_stats_output/qft_4q_gs_prec5_stats.json \
  --counts pbc \
  --model 'majorana 1e-6'
```

Because FTCircuitBench has already synthesised every `rz` into Clifford+T, the
counts handed to QRE carry a concrete T count and leave QRE's own rotation-cost
model unused: the physical estimate reflects *your* choice of synthesis engine
and precision. See [`docs/examples.md`](docs/examples.md#4-physical-resource-estimation-estimate_resourcespy)
for the programmatic API.

## Repository structure

```
FTCircuitBench/
├── ftcircuitbench/                     # Library (API, analyzers, transpilers, PBC converter)
│   ├── api.py                          # Public entry points: run_pipeline, run_analysis*
│   ├── analyzer/                       # Clifford+T and PBC circuit analyzers
│   ├── decomposer/                     # Gate decomposition utilities
│   ├── parser/                         # QASM parser
│   ├── pbc_converter/                  # PBC circuit conversion and I/O
│   ├── transpilers/                    # Gridsynth and Solovay-Kitaev transpilers
│   ├── resource_estimation/            # Azure QRE bridge (optional `qre` extra)
│   ├── frontends/                      # Cirq / Qualtran ingest (optional extras)
│   └── reports/                        # Markdown summary generation
├── analyze_circuit.py                  # CLI: analyze a single circuit
├── generate_benchmarks.py              # CLI: run the full benchmark suite
├── estimate_resources.py               # CLI: physical resource estimates via Azure QRE
├── import_circuit.py                   # CLI: import a Cirq/Qualtran circuit as QASM
├── tools/export_cirq_qasm.py           # Standalone Cirq->QASM exporter (pyLIQTR envs)
├── demos/                              # Executable demo notebooks (committed with outputs)
│   ├── Qualtran_to_QRE_Demo.ipynb      # Qualtran -> FTCB -> Azure QRE in four steps
│   └── FTCircuitBench_Pipeline_Demo.ipynb  # The compilation pipeline in depth
├── qasm/                               # Input benchmark circuits (QASM 2.0)
├── circuit_outputs/                    # Archival Clifford+T QASM artifacts (legacy backend)
├── circuit_stats_output/               # Sample output statistics (JSON)
├── figs/                               # Reference output figures (PDF)
├── tests/                              # pytest test suite
├── docs/                               # API reference, installation guide, examples
└── pyproject.toml
```

## Documentation

- [`docs/index.md`](docs/index.md) — documentation entry point
- [`docs/installation.md`](docs/installation.md) — detailed setup instructions
- [`docs/api.md`](docs/api.md) — public Python API reference
- [`docs/examples.md`](docs/examples.md) — CLI and programmatic recipes

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to set up a development environment, run tests, and submit changes.

## Citation

If you use FTCircuitBench in your research, please cite:

```bibtex
@misc{harkness2026ftcircuitbenchbenchmarksuitefaulttolerant,
      title={FTCircuitBench: A Benchmark Suite for Fault-Tolerant Quantum Compilation and Architecture},
      author={Adrian Harkness and Shuwen Kan and Chenxu Liu and Meng Wang and John M. Martyn and Shifan Xu and Diana Chamaki and Ethan Decker and Ying Mao and Luis F. Zuluaga and Tamás Terlaky and Ang Li and Samuel Stein},
      year={2026},
      eprint={2601.03185},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2601.03185},
}
```

See also `CITATION.cff` for machine-readable metadata.

## Troubleshooting

**`gridsynth` binary not found**

The `gridsynth` Haskell binary enables the Python-fallback Gridsynth path; the C++ `nwqec` backend is preferred and used automatically when available. If you need the Haskell `gridsynth`, install it via `cabal install newsynth` (the executable is named `gridsynth` but lives in the `newsynth` package on Hackage) and ensure the cabal bin directory is on your `PATH`.

**`nwqec` install fails**

`nwqec` is a binary wheel for the C++ Clifford+T / PBC transpilers. If your platform doesn't have a prebuilt wheel, see [`docs/installation.md`](docs/installation.md) for build-from-source instructions. The pure-Python fallbacks (`gs_transpiler`, `sk_transpiler`) work without `nwqec`.

**Long benchmark runs**

`generate_benchmarks.py` exercises every circuit in `qasm/` and can take hours. To run a subset, see the CLI flags in [`docs/examples.md`](docs/examples.md) or the `analyze_circuit.py` per-circuit invocation.
