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
- Native tableau ops for `sdg`, `x`, `y`, `z` in
  `pbc_converter.tab_gate.TableauForGate`. Sdg is now applied directly instead
  of being decomposed into three S gates; X/Y/Z were previously rejected as
  unsupported and are now first-class Cliffords in the converter.
- `transpilers._basis` module: shared canonical constants
  (`INTERMEDIATE_RZ_BASIS`, `PBC_COMPATIBLE_CLIFFORD_T_BASIS`) and helpers
  (`prepare_input`, `to_intermediate_rz`, `enforce_pbc_basis`,
  `is_clifford_t_basis`) so the GS / SK / NWQEC pipelines all share the same
  staging — only the RZ-synthesis step differs.
- `--gs-backend {auto,cpp,python}` CLI flag on `analyze_circuit.py` for
  explicit selection between the nwqec C++ Gridsynth backend and the Python
  Gridsynth implementation. Default `auto` preserves prior behaviour
  (prefer cpp when nwqec is installed).
- Test modules `tests/test_tab_gate_paulis.py`,
  `tests/test_pbc_extended_basis.py`, and an expanded
  `tests/test_pipeline_parity.py` covering tableau correctness, PBC basis
  acceptance, and behavioural parity with NWQEC C++ as the reference
  implementation.

### Changed

- License metadata aligned to MIT across the project. Previously `pyproject.toml`
  declared Apache-2.0 while `LICENSE` was MIT; the canonical license is MIT.
- Install path is now uv-first (`uv sync --all-extras`); pip remains a documented
  fallback in `README.md` and `docs/installation.md`.
- Regenerated `docs/api.md` against the actual public API surface exported from
  `ftcircuitbench.api`.
- Expanded `docs/examples.md` with additional CLI and programmatic recipes.
- **Broadened the canonical Clifford+T basis** end-to-end. The intermediate
  basis used by `to_intermediate_rz` is now
  `{cx, h, s, sdg, t, tdg, x, y, z, rz}` (was `{cx, h, s, rz}`), and the
  PBC-compatible output basis is `{cx, h, s, sdg, t, tdg, x, y, z}` (was
  `{cx, h, s, t, tdg}`). The Solovay-Kitaev approximation basis was widened
  to match. Discrete Clifford+T gates in the input now flow through synthesis
  unchanged instead of being rewritten through `rz` and re-approximated. Python
  GS now matches NWQEC C++ behaviorally on inputs containing `{tdg, sdg, x, y, z}`.
- DRY refactor of `RotationPauliCirc.process` in
  `pbc_converter/r_pauli_circ.py`: the previously duplicated
  `ifprint=True/False` branches now share a single loop body, with `tqdm`
  applied conditionally to the iterator.

### Removed

- Python 3.9 support. The minimum supported Python version is now 3.10.
- `"v3"` PBC layering method. It was a thin alias for `"v2"` with
  `layering_max_checks` set; the same behaviour is now available by passing
  `layering_method="v2"` together with `layering_max_checks=K`. The `Literal`
  type, `--layering-method` CLI choice, and `TableauPauliBasis.layer_v3` method
  have all been dropped. **Breaking change** for callers that explicitly passed
  `layering_method="v3"`.

### Performance

- `TableauPauliBasis.layer_v2` now hoists the X/Z swap of the candidate Pauli
  out of the per-layer commutation check, and skips layers whose qubit support
  has no overlap with the candidate's support (tracked via per-layer integer
  bitmasks). Together these changes reduce wall-clock time on the layering
  benchmark by ~15% overall, with up to ~40% reductions on sparse circuits and
  ~10% reductions on the full `optimize_t` macro-bench. No behavioural change
  to the layered output.

### Fixed

- 149 mypy errors across the codebase. Two `pbc_converter` modules
  (`pbc_circuit_reader`, `pbc_generator`) carry deeper type issues that require
  an API/data-model review and are deferred via documented per-module
  `[[tool.mypy.overrides]]` entries in `pyproject.toml`.

## [0.1.0] - 2026-04-05

### Added

- Initial release accompanying [arXiv:2601.03185](https://arxiv.org/abs/2601.03185).
