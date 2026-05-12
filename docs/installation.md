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
git clone https://github.com/AdrianHarkness/FTCircuitBench.git
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
git clone https://github.com/AdrianHarkness/FTCircuitBench.git
cd FTCircuitBench

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

Drop `[dev]` if you don't need pytest, ruff, black, or isort.

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
