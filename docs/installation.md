# Installation

Tested on Python 3.10+. The recommended setup uses [uv](https://docs.astral.sh/uv/)
together with the committed `uv.lock` for reproducible installs.

## Prerequisites

- Python 3.10+ (the project pins `3.11` via `.python-version`; uv will fetch a
  matching interpreter automatically if one isn't already available).
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or
  a virtual environment tool (`python -m venv`) plus `pip` for the fallback path.
  If a Conda env is active, deactivate it first.
- Optional: Gridsynth binary on your `PATH` (install via `cabal install gridsynth`
  and add the cabal bin dir). The package prefers the binary over the Python
  wrapper.

All runtime dependencies are declared in `pyproject.toml`. With uv they are
locked in `uv.lock`; with pip they are resolved fresh on install.

## Recommended: uv

```bash
git clone https://github.com/AdrianHarkness/FTCircuitBench.git
cd FTCircuitBench

uv sync --all-extras       # creates .venv with all deps + dev tools
```

`uv sync` creates `.venv/` in the repo root and installs the project in editable
mode using the pinned versions from `uv.lock`. To run commands inside that
environment, prefix them with `uv run`:

```bash
uv run pytest
uv run python analyze_circuit.py --help
```

If you'd rather activate the venv directly, `source .venv/bin/activate` works
too (Windows: `.venv\Scripts\activate`).

## pip (alternative)

If you prefer pip and a manually managed virtual environment:

```bash
git clone https://github.com/AdrianHarkness/FTCircuitBench.git
cd FTCircuitBench

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

(Drop `[dev]` if you don't need pytest/ruff/black/isort.)

## Quick checks

```bash
uv run python analyze_circuit.py --help
uv run python - <<'PY'
from ftcircuitbench.api import PipelineConfig, run_analysis_for_file
print("OK: ftcircuitbench import")
PY
```

## Optional: gridsynth binary

If the Gridsynth binary is missing, the Python path still works but runs
slower; keep the binary available for best performance.

## Notes on platform support

`nwqec` ships prebuilt wheels for the common platforms (recent macOS and Linux
on x86_64/arm64) for Python 3.10–3.12. If `uv sync` cannot find a wheel for
your platform/interpreter combination, fall back to the pip path above with a
locally built `nwqec`, or open an issue.
