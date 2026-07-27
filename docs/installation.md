# Installation

FTCircuitBench targets Python 3.10+ and is developed against Python 3.11
(pinned via `.python-version`). The recommended setup uses
[uv](https://docs.astral.sh/uv/) together with the committed `uv.lock` for
reproducible installs; a pip-based fallback is documented below.

## Prerequisites

- Python 3.10+. uv will fetch a matching interpreter automatically when one
  isn't already available.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended),
  or `python -m venv` plus `pip` for the fallback path. If a Conda
  environment is active, deactivate it before running uv or pip.
- Optional: the Haskell `gridsynth` binary on `PATH`. See
  [Optional: gridsynth binary](#optional-gridsynth-binary) below.

All runtime dependencies are declared in `pyproject.toml`; uv pins them in
`uv.lock`, while pip resolves them fresh on install.

## Recommended: uv

```bash
git clone https://github.com/pnnl/FTCircuitBench.git
cd FTCircuitBench

uv sync --all-extras       # creates .venv with all deps + dev tools
```

`uv sync` creates `.venv/` in the repository root and installs FTCircuitBench
in editable mode using the pinned versions from `uv.lock`. To run commands
inside that environment, prefix them with `uv run`:

```bash
uv run pytest
uv run python analyze_circuit.py --help
```

If you'd rather activate the venv directly, `source .venv/bin/activate` works
(Windows: `.venv\Scripts\activate`).

## pip (alternative)

If you prefer pip and a manually managed virtual environment:

```bash
git clone https://github.com/pnnl/FTCircuitBench.git
cd FTCircuitBench

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev,qre]"
```

Drop `[dev]` if you don't need pytest, ruff, or mypy, and `[qre]` if you don't
need the Azure resource-estimation bridge (see below).

## Optional: Azure Quantum Resource Estimator

The `qre` extra installs [`qdk`](https://pypi.org/project/qdk/), which bundles
the Azure Quantum Resource Estimator used by `estimate_resources.py` and
`ftcircuitbench.resource_estimation`. The estimator runs locally — no Azure
subscription, account, or network access is involved.

```bash
uv sync --extra qre         # or: pip install "ftcircuitbench[qre]"
```

`uv sync --all-extras` already includes it. Without the extra, everything else
in the suite works normally; the estimation CLI exits with an install hint and
the estimation tests that need it are skipped.

Note that `qdk` supersedes the older `qsharp` package. The bridge imports
`qdk.estimator` first and falls back to `qsharp.estimator`, so an environment
that still has the old package keeps working.

## Optional: Cirq and Qualtran frontends

The `cirq` extra installs `cirq-core`, which lets `ftcircuitbench.frontends`
ingest circuits from any Cirq-based generator. The `qualtran` extra adds
Qualtran itself (and pulls `cirq-core` transitively), which lets a Bloq go
straight into the pipeline:

```bash
uv sync --extra qualtran        # or: pip install "ftcircuitbench[qualtran]"
```

`uv sync --all-extras` includes both. Qualtran pulls a sizeable dependency tree
(including Jupyter components it uses for its own notebooks), so install just
`--extra cirq` if all you need is to read Cirq circuits or QASM files.

### pyLIQTR is a separate environment

pyLIQTR releases pin `numpy<2` and `qualtran==0.4.0`, neither of which can
coexist with FTCircuitBench's dependency floor (`numpy>=2.0.0`), so there is no
extra for it. Export QASM from a pyLIQTR environment with
`tools/export_cirq_qasm.py` — a standalone script that imports nothing from
`ftcircuitbench` — and analyse the file here. See
[`examples.md`](examples.md#pyliqtr).

## Quick checks

After install, verify both the CLI and the library import cleanly:

```bash
uv run python analyze_circuit.py --help
uv run python - <<'PY'
from ftcircuitbench.api import PipelineConfig, run_analysis_for_file
print("OK: ftcircuitbench import")
PY
```

For an end-to-end smoke test (a 4-qubit QFT through the Gridsynth + PBC
pipeline) see the README's "Reproducing paper results" section and the
single-circuit walkthrough in [`examples.md`](examples.md).

## Optional: gridsynth binary

FTCircuitBench's Clifford+T synthesis prefers the `nwqec` C++ backend (shipped
as a binary wheel and pulled in automatically by `uv sync` / `pip install`).
On platforms without a prebuilt `nwqec` wheel, the package falls back to a
pure-Python Gridsynth path that shells out to the Haskell `gridsynth` CLI.

Install the Haskell `gridsynth` binary if you need that fallback path:

```bash
cabal install gridsynth
# Then make sure ~/.cabal/bin (or your cabal bin dir) is on PATH.
```

Once the binary is on `PATH` it is detected automatically; the API does not
need to be re-configured.

## Notes on platform support

`nwqec` ships prebuilt wheels for recent macOS and Linux on x86_64 and arm64
for Python 3.10–3.12. If `uv sync` cannot find a wheel for your
platform/interpreter combination, fall back to the pip path above with a
locally built `nwqec`, or open an issue.

## Next steps

- [`examples.md`](examples.md) — CLI and Python recipes.
- [`api.md`](api.md) — public API reference.
- [`index.md`](index.md) — documentation entry point.
