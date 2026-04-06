# FTCircuitBench

[![arXiv](https://img.shields.io/badge/arXiv-2601.03185-b31b1b.svg)](https://arxiv.org/abs/2601.03185)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

A benchmark suite for fault-tolerant quantum circuit compilation and architecture, covering Clifford+T synthesis (Gridsynth and Solovay-Kitaev) and Pauli-Based Computation (PBC).

## Install

```bash
git clone https://github.com/AdrianHarkness/FTCircuitBench.git
cd FTCircuitBench
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requirements: Python 3.9+, [`nwqec`](https://github.com/pnnl/nwqec) (for fast Gridsynth/PBC via `fuse_t`). An optional `gridsynth` binary on your `PATH` enables the Python-fallback GS path.

## Quick start

Analyze one circuit (Gridsynth pipeline, PBC on):

```bash
python analyze_circuit.py qasm/qft/qft_18q.qasm \
  --pipeline gs \
  --gridsynth-precision 5
```

Run the full benchmark suite:

```bash
python generate_benchmarks.py
```

Open the walkthrough notebook:

```bash
jupyter notebook FTCircuitBench_Pipeline_Demo.ipynb
```

Select the project `.venv` kernel and run all cells.

**Common CLI flags:** `--pipeline {gs,sk,both}`, `--gridsynth-precision N`, `--sk-recursion N`, `--layering-max-checks K`, `--optimize-pbc`, `--optimize-t-maxiter N`, `--skip-fidelity`, `--max-workers N`.

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
├── circuit_stats_output/               # Sample output statistics (JSON)
├── figs/                               # Reference output figures (PDF)
├── tests/                              # pytest test suite
├── docs/                               # API reference, installation guide, examples
└── pyproject.toml
```

## Documentation

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
