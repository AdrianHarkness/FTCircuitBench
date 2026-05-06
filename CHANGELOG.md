# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Pre-commit configuration (`.pre-commit-config.yaml`) covering ruff, black,
  isort, and assorted hygiene hooks.
- `mypy` configuration in `pyproject.toml` and a `[tool.coverage]` section for
  `pytest-cov`.
- CI matrix exercising Python 3.10, 3.11, and 3.12, plus separate
  lint, typecheck, and coverage CI jobs.
- `uv.lock` checked in for reproducible installs via `uv sync`.
- `.python-version` pinning the project interpreter (3.11).
- `CHANGELOG.md` (this file) and `CITATION.cff` for repository archaeology and
  machine-readable citation metadata.
- "Reproducing paper results" and "Troubleshooting" sections in `README.md`.

### Changed

- License metadata aligned to MIT across the project. Previously `pyproject.toml`
  declared Apache-2.0 while `LICENSE` was MIT; the canonical license is MIT.
- Install path is now uv-first (`uv sync --all-extras`); pip remains a documented
  fallback in `README.md` and `docs/installation.md`.
- Regenerated `docs/api.md` against the actual public API surface exported from
  `ftcircuitbench.api`.
- Expanded `docs/examples.md` with additional CLI and programmatic recipes.

### Removed

- Python 3.9 support. The minimum supported Python version is now 3.10.

### Fixed

- 149 mypy errors across the codebase. Two `pbc_converter` modules
  (`pbc_circuit_reader`, `pbc_generator`) carry deeper type issues that require
  an API/data-model review and are deferred via documented per-module
  `[[tool.mypy.overrides]]` entries in `pyproject.toml`.

## [0.1.0] - 2026-04-05

### Added

- Initial release accompanying [arXiv:2601.03185](https://arxiv.org/abs/2601.03185).
