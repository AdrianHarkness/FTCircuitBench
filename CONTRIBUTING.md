# Contributing to FTCircuitBench

Thank you for your interest in contributing! This document covers how to set up a development environment, run tests, and submit changes.

## Development setup

```bash
git clone https://github.com/AdrianHarkness/FTCircuitBench.git
cd FTCircuitBench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the package in editable mode along with all development dependencies (pytest, ruff, black, isort).

## Running tests

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Some tests require `nwqec` to be installed and (optionally) a `gridsynth` binary on your `PATH`. Tests that depend on optional tooling are skipped automatically when those tools are unavailable.

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting, [black](https://black.readthedocs.io/) for formatting, and [isort](https://pycqa.github.io/isort/) for import ordering. All three are configured in `pyproject.toml`.

Check and auto-fix before committing:

```bash
ruff check --fix ftcircuitbench/ tests/
black ftcircuitbench/ tests/
isort ftcircuitbench/ tests/
```

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes, add tests where appropriate.
3. Ensure `pytest` passes and the style checks above produce no errors.
4. Open a pull request against `main` with a clear description of what changed and why.

## Reporting issues

Please open a [GitHub issue](https://github.com/AdrianHarkness/FTCircuitBench/issues) and include:

- A minimal reproducible example (QASM file and the command or code that triggers the issue).
- The Python version, OS, and version of `nwqec` / `gridsynth` in use.
- The full error traceback.
