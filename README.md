# FTCircuitBench

[![CI](https://github.com/AdrianHarkness/FTCircuitBench/actions/workflows/ci.yml/badge.svg)](https://github.com/AdrianHarkness/FTCircuitBench/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2601.03185-b31b1b.svg)](https://arxiv.org/abs/2601.03185)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A benchmark suite for fault-tolerant quantum circuit compilation and architecture, covering Clifford+T synthesis (Gridsynth and Solovay-Kitaev) and Pauli-Based Computation (PBC).

## Install

```bash
git clone https://github.com/AdrianHarkness/FTCircuitBench.git
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

Open the walkthrough notebook:

```bash
jupyter notebook FTCircuitBench_Pipeline_Demo.ipynb
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

The values above are the canonical reference for this smoke test on the current `nwqec` C++ backend. If your numbers match (or are within rounding for `_count` totals), your install is reproducing the pipeline correctly. The legacy QASM artifacts under `circuit_outputs/` were produced by an older Python-Gridsynth path and predate the C++ backend; they are kept for archival reference but should not be used as a head-to-head comparison.

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
│   └── reports/                        # Markdown summary generation
├── analyze_circuit.py                  # CLI: analyze a single circuit
├── generate_benchmarks.py              # CLI: run the full benchmark suite
├── FTCircuitBench_Pipeline_Demo.ipynb  # Walkthrough notebook
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
