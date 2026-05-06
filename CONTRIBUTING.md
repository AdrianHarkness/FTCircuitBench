# Contributing to FTCircuitBench

Thank you for your interest in contributing! This document covers how to set up a development environment, run tests, and submit changes.

## Development setup

The recommended workflow uses [uv](https://docs.astral.sh/uv/) and the
committed `uv.lock` for reproducible installs. The pinned interpreter is read
from `.python-version` (currently `3.11`).

```bash
git clone https://github.com/AdrianHarkness/FTCircuitBench.git
cd FTCircuitBench
uv sync --all-extras
```

This creates `.venv/` and installs the package in editable mode along with all
development dependencies (pytest, pytest-mock, ruff, black, isort) using the
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

## Running tests

```bash
uv run pytest
```

Run with verbose output:

```bash
uv run pytest -v
```

Some tests require `nwqec` to be installed and (optionally) a `gridsynth` binary on your `PATH`. Tests that depend on optional tooling are skipped automatically when those tools are unavailable.

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting, [black](https://black.readthedocs.io/) for formatting, and [isort](https://pycqa.github.io/isort/) for import ordering. All three are configured in `pyproject.toml`.

Check and auto-fix before committing:

```bash
uv run ruff check --fix ftcircuitbench/ tests/
uv run black ftcircuitbench/ tests/
uv run isort ftcircuitbench/ tests/
```

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes, add tests where appropriate.
3. Ensure `uv run pytest` passes and the style checks above produce no errors.
4. Open a pull request against `main` with a clear description of what changed and why.

## Reporting issues

Please open a [GitHub issue](https://github.com/AdrianHarkness/FTCircuitBench/issues) and include:

- A minimal reproducible example (QASM file and the command or code that triggers the issue).
- The Python version, OS, and version of `nwqec` / `gridsynth` in use.
- The full error traceback.
