# Examples

Minimal ways to run the pipelines from the repo checkout.

## CLI helper (`analyze_circuit.py`)

Run GS and SK with PBC optimization and save artifacts:

```bash
python analyze_circuit.py qasm/hhl/hhl_7q.qasm \
  --pipeline both \
  --gridsynth-precision 4 \
  --sk-recursion 2 \
  --optimize-pbc
```

Outputs land in:
- `clifford_t_output/` — Clifford+T QASM.
- `pbc_output/` — PBC layer + measurement basis text files.
- `circuit_stats_output/` — JSON summaries.

See `python analyze_circuit.py --help` for all flags (e.g., `--skip-fidelity`, `--max-workers`).

## Programmatic usage

```python
from ftcircuitbench.api import PipelineConfig, run_analysis_for_file

cfgs = [
    PipelineConfig(pipeline="gs", gridsynth_precision=4, optimize_pbc=True),
    PipelineConfig(pipeline="sk", sk_recursion=2, calculate_fidelity=False),
]
analysis = run_analysis_for_file("qasm/hhl/hhl_7q.qasm", cfgs)
print(analysis.pipelines["gs"].pbc_stats["pbc_rotation_operators"])
```

## Batch + notebook

- `generate_benchmarks.py`: sweeps over circuits in `qasm/`; run `python generate_benchmarks.py --help`.
- `FTCircuitBench_Pipeline_Demo.ipynb`: a step-by-step, notebook-style walkthrough.
