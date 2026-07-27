# Contributing to FTCircuitBench

Thank you for your interest in contributing! This document covers how to set up a development environment, run tests, and submit changes.

## Development setup

The recommended workflow uses [uv](https://docs.astral.sh/uv/) and the
committed `uv.lock` for reproducible installs. The pinned interpreter is read
from `.python-version` (currently `3.11`).

```bash
git clone https://github.com/pnnl/FTCircuitBench.git
cd FTCircuitBench
uv sync --all-extras
```

This creates `.venv/` and installs the package in editable mode along with all
development dependencies (pytest, pytest-mock, ruff, mypy) using the
pinned versions in `uv.lock`. Run any project command with `uv run`:

```bash
uv run pytest
uv run ruff check ftcircuitbench/ tests/
```

### pip (fallback)

If you'd rather use pip and a manually managed virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### gridsynth (optional)

`gridsynth` is a separate, non-Python tool used by the Python Rz-synthesis path
(`ftcircuitbench.decomposer`, which shells out to the binary). It is not
installable via `uv`/`pip`. Without it, the ~20 parity tests that exercise that
path skip; everything else, including the nwqec C++ backend, works normally.

It ships in Peter Selinger's Haskell [`newsynth`](https://hackage.haskell.org/package/newsynth)
package. CI builds it the same way (see `.github/workflows/ci.yml`):

```bash
cabal update
cabal install newsynth --install-method=copy --overwrite-policy=always \
  --installdir="$HOME/.cabal/bin"
export PATH="$HOME/.cabal/bin:$PATH"   # add to your shell rc to persist
```

**GHC version constraint.** `newsynth` builds via a custom `Setup.hs` that
depends on `superdoc`, which uses the pre-3.12 `FilePath`-based Cabal API. It
does **not** compile with GHC 9.10+, where that API became `SymbolicPath` — the
build fails in `Distribution/Superdoc/Hooks.hs` with `Couldn't match type
[Char] with ... SymbolicPath`. CI pins GHC 9.6, so install a matching compiler
rather than whatever your package manager ships by default:

```bash
# macOS/Linux; ghcup is also available via brew, apt, or the ghcup installer
ghcup install ghc 9.6.6
cabal install newsynth -w "$HOME/.ghcup/bin/ghc-9.6.6" \
  --install-method=copy --overwrite-policy=always --installdir="$HOME/.cabal/bin"
```

Verify with:

```bash
gridsynth "(0.3)" --digits=3   # prints an H/T/S gate string
```

## Running tests

```bash
uv run pytest
```

Run with verbose output:

```bash
uv run pytest -v
```

Some tests require `nwqec` to be installed and (optionally) a `gridsynth` binary
on your `PATH` (see [gridsynth (optional)](#gridsynth-optional) above). Tests
that depend on optional tooling are skipped automatically when those tools are
unavailable, so a run without them should still be green — just with more
skips.

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting, formatting,
and import ordering, configured under `[tool.ruff]` in `pyproject.toml`.

Check and auto-fix before committing:

```bash
uv run ruff check --fix ftcircuitbench/ tests/
uv run ruff format ftcircuitbench/ tests/
```

## Pre-commit hooks

This repository ships a [pre-commit](https://pre-commit.com/) configuration
(`.pre-commit-config.yaml`) that runs ruff (lint + format) and a few standard
hygiene checks (trailing whitespace, end-of-file newline, YAML/TOML syntax,
large-file guard) on every commit. Installing the hooks is optional but
recommended:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

The first command installs the git hook in your local clone; the second runs
all hooks against the entire repository so you can verify a clean baseline.

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes, add tests where appropriate.
3. Ensure `uv run pytest` passes and the style checks above produce no errors.
4. Open a pull request against `main` with a clear description of what changed and why.

## Reporting issues

Please open a [GitHub issue](https://github.com/pnnl/FTCircuitBench/issues) and include:

- A minimal reproducible example (QASM file and the command or code that triggers the issue).
- The Python version, OS, and version of `nwqec` / `gridsynth` in use.
- The full error traceback.
