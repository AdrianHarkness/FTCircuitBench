# FTCircuitBench documentation

Fault-tolerant circuit compilation and analysis with Gridsynth (GS),
Solovay-Kitaev (SK), and Pauli-Based Computation (PBC).

## Documentation pages

- [`installation.md`](installation.md) — install via uv (recommended) or pip,
  with notes on the optional `gridsynth` Haskell binary.
- [`api.md`](api.md) — reference for every symbol re-exported from
  `ftcircuitbench.__init__` and the entry points in `ftcircuitbench.api`.
- [`examples.md`](examples.md) — five worked examples: single-circuit CLI,
  batch CLI, the Python API, physical resource estimation, and importing
  circuits from Qualtran and pyLIQTR.

## Other useful links

- [`../README.md`](../README.md) — top-level overview and quick start.
- [`../README.md#reproducing-paper-results`](../README.md#reproducing-paper-results)
  — smoke-test and full benchmark reproduction commands.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — development setup, tests, and
  contribution guidelines.
- `demos/FTCircuitBench_Pipeline_Demo.ipynb` — annotated notebook
  walkthrough of the pipeline.

## Quick orientation

- FTCircuitBench is the middle layer: Qualtran / pyLIQTR construct logical
  circuits, FTCircuitBench compiles and characterises them, Azure QRE prices the
  physical machine. `demos/Qualtran_to_QRE_Demo.ipynb` walks that
  path end to end.
- Pipelines: GS or SK → Clifford+T → PBC conversion → optional fidelity + stats,
  then optionally down to physical costs via the Azure Resource Estimator.
- Output directories: `clifford_t_output/` (Clifford+T QASM),
  `pbc_output/` (PBC layers and measurement bases),
  `circuit_stats_output/` (JSON summaries), and `qre_output/` (physical
  resource estimates).
- Scripts: `analyze_circuit.py` for a single circuit, `generate_benchmarks.py`
  for the full sweep, `estimate_resources.py` for physical resource estimates,
  `import_circuit.py` for Cirq/Qualtran ingest.
- Library: import `PipelineConfig`, `run_pipeline`, or `run_analysis_for_file`
  from `ftcircuitbench.api`; `ftcircuitbench.resource_estimation` for the Azure
  QRE bridge (optional `qre` extra); `ftcircuitbench.frontends` for Cirq and
  Qualtran ingest (optional `cirq` / `qualtran` extras).

Start with [`installation.md`](installation.md), then try
[`examples.md`](examples.md).
