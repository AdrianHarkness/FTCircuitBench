# FTCircuitBench documentation

Fault-tolerant circuit compilation and analysis with Gridsynth (GS),
Solovay-Kitaev (SK), and Pauli-Based Computation (PBC).

## Documentation pages

- [`installation.md`](installation.md) — install via uv (recommended) or pip,
  with notes on the optional `gridsynth` Haskell binary.
- [`api.md`](api.md) — reference for every symbol re-exported from
  `ftcircuitbench.__init__` and the entry points in `ftcircuitbench.api`.
- [`examples.md`](examples.md) — three worked examples: single-circuit CLI,
  batch CLI, and the Python API.

## Other useful links

- [`../README.md`](../README.md) — top-level overview and quick start.
- [`../README.md#reproducing-paper-results`](../README.md#reproducing-paper-results)
  — smoke-test and full benchmark reproduction commands.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — development setup, tests, and
  contribution guidelines.
- `FTCircuitBench_Pipeline_Demo.ipynb` (repo root) — annotated notebook
  walkthrough of the pipeline.

## Quick orientation

- Pipelines: GS or SK → Clifford+T → PBC conversion → optional fidelity + stats.
- Output directories: `clifford_t_output/` (Clifford+T QASM),
  `pbc_output/` (PBC layers and measurement bases), and
  `circuit_stats_output/` (JSON summaries).
- Scripts: `analyze_circuit.py` for a single circuit, `generate_benchmarks.py`
  for the full sweep.
- Library: import `PipelineConfig`, `run_pipeline`, or `run_analysis_for_file`
  from `ftcircuitbench.api`.

Start with [`installation.md`](installation.md), then try
[`examples.md`](examples.md).
