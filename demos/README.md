# Demos

Two executable notebooks, both committed with their outputs so they can be read
without running anything.

| Notebook | What it shows | Runtime |
|---|---|---|
| [`Qualtran_to_QRE_Demo.ipynb`](Qualtran_to_QRE_Demo.ipynb) | The full stack in four steps: a Qualtran Bloq → FTCircuitBench compilation → Azure QRE physical qubits and runtime. Start here. | seconds |
| [`FTCircuitBench_Pipeline_Demo.ipynb`](FTCircuitBench_Pipeline_Demo.ipynb) | FTCircuitBench's own pipeline in depth on a 100-qubit Hamiltonian circuit: Clifford+T synthesis, PBC conversion, and every analysis plot. | ~45 min (seconds with a smaller input; see its intro) |

## Running them

From the repository root:

```bash
uv sync --all-extras
uv run jupyter notebook demos/
```

and select the project `.venv` kernel. The notebooks resolve paths relative to
the repository root, so they can be launched from here or from the root.
