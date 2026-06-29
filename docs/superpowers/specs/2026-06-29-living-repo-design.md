# LPBF-Optimizer Living Repository Design

**Date:** 2026-06-29  
**Status:** Draft — pending review  
**Scope:** Transform the LPBF-Optimizer research prototype into a maintainable, documented, tested, and correctly-running repository with a clear research roadmap.

## 1. Goal

Make LPBF-Optimizer a "living repo":

- A new contributor can clone, install dependencies, run the end-to-end workflow, and run tests in under 30 minutes.
- Documentation matches the code.
- Known runtime bugs are fixed.
- The repository follows common open-source standards (LICENSE, CONTRIBUTING, CHANGELOG, issue templates, CI).
- Key architectural and scientific trade-offs are recorded in Architecture Decision Records (ADRs).
- A research-backed roadmap guides future work.

## 2. Scope and non-scope

### In scope

1. **Correctness fixes** in core training and optimization code.
2. **Configuration cleanup** (relative paths, missing fields, dependency fixes).
3. **Documentation refresh** (README, AGENTS, workflow docs, equations, notebooks).
4. **Testing infrastructure** (`tests/`, fixtures, CI).
5. **Repository standards** (LICENSE, CONTRIBUTING, CHANGELOG, issue templates).
6. **Developer experience** (`pyproject.toml`, `environment.yml`, pre-commit hooks, script path robustness).
7. **Research planning** (ADRs, literature survey, updated roadmap).

### Out of scope

- Rewriting the FEA runner to work without ABAQUS/COMSOL licenses.
- Implementing real-time closed-loop control.
- Replacing stub validation modules with production machine interfaces.
- Changing the public input/output dimensions of the PINN.

These remain stub/placeholder modules, but their limitations must be clearly documented and warned about at runtime.

## 3. Decomposition into workstreams

| Workstream | Main deliverables | Files touched |
|------------|-------------------|---------------|
| W1. Foundation & correctness | Fix runtime bugs; make `generate` → `train` → `optimize` run end-to-end. | `src/pinn/physics.py`, `src/pinn/train.py`, `src/pinn/model.py`, `src/optimiser/nsga3.py`, `src/optimiser/bayesopt.py`, `data/params.yaml`, `requirements.txt` |
| W2. Documentation refresh | Accurate README, AGENTS, workflow docs, equations, fix orphan docs. | `README.md`, `SUBMISSION.md`, `AGENTS.md`, `docs/*.md`, `notebooks/*.ipynb` |
| W3. Testing & quality | `tests/` mirroring `src/`, passing `pytest`, linting. | `tests/**/*.py`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml` |
| W4. Developer experience | Installable package, Conda env, contribution guide, issue templates. | `pyproject.toml`, `environment.yml`, `LICENSE`, `CONTRIBUTING.md`, `CHANGELOG.md`, `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md` |
| W5. Research & planning | ADRs, literature survey, updated roadmap. | `docs/adr/*.md`, `docs/research/*.md`, `todo.md` |

## 4. Detailed design

### 4.1 Foundation & correctness (W1)

#### 4.1.1 Physics loss reconciliation

**Problem:** `src/pinn/physics.py` assumes the PINN outputs are `temperature` and `stress tensor components`, but the actual model outputs are `residual_stress`, `porosity`, and `geometric_accuracy`.

**Decision:** Keep the public 3-output interface and derive *physics-informed regularization residuals* from the quality-metric outputs. Because the outputs are empirical quality metrics rather than direct physical fields, the residuals are simplified, physically-motivated penalty terms rather than full PDEs.

- `residual_stress` (index 0) → stress-equilibrium residual computed from the gradient of the predicted residual-stress field, plus a thermo-elastic body-force term derived from the analytic temperature field.
- `porosity` (index 1) → porosity-formation residual that penalizes large deviations between the predicted porosity and an empirical porosity indicator based on energy density and solidification conditions.
- `geometric_accuracy` (index 2) → geometry-conservation residual that penalizes predicted geometric distortion inconsistent with the nominal build geometry and thermal strain field.

The existing heat-equation residual will be preserved as an auxiliary residual computed from a derived temperature field `T` inferred from the laser parameters and coordinates (Rosenthal-like analytic field), not from a network output. This keeps the network architecture unchanged while still penalizing physically inconsistent predictions.

#### 4.1.2 Objective direction in NSGA-III

**Problem:** `src/optimiser/nsga3.py` minimizes all three objectives, but `geometric_accuracy` should be maximized.

**Fix:** Negate `geometric_accuracy` inside `_evaluate` so NSGA-III minimizes `-accuracy`. Document this clearly in code comments and docs.

#### 4.1.3 Configuration and paths

- Convert all absolute Windows paths in `data/params.yaml` to relative paths.
- Add `repo_root` resolution helper in scripts so they work when run from any directory.
- Add missing fields: `validate:` section, `random_seed`, `device`, normalization flags.
- Standardize `scheduler.type` to `reduce_lr_on_plateau` in both `data/params.yaml` and `src/pinn/train.py`.
- Set `n_epochs` default to a value useful for a real smoke test (e.g., 50) and document that 2 epochs is only for CI.

#### 4.1.4 Dependency fixes

- Add `seaborn` to `requirements.txt` (used by `train.py` and `vis`).
- Add `scikit-learn` to `requirements.txt` (used by `characterise.py`).
- Review `ax-platform` API usage in `bayesopt.py` and pin compatible versions or refactor calls.

#### 4.1.5 Code quality fixes

- Remove duplicate `from model import PINN` in `src/pinn/train.py`.
- Remove leftover `DEBUG` prints in `train.py`.
- Remove duplicate `import os` in `nsga3.py`.
- Remove unused Ax imports in `bayesopt.py`.
- Replace broad `except Exception` around loss balancer with targeted handling.
- Refactor duplicated input-trimming logic in `train.py` into a helper.

### 4.2 Documentation refresh (W2)

- Rewrite `README.md` Quick Start to include data generation first.
- Fix `docs/pinn_model_architecture.md` input dimension to 10 and epochs to match config.
- Fix `docs/training_metrics.md` unrendered template variables.
- Remove or rewrite `docs/figures.md` (currently unrelated defect-detection content).
- Label all GIFs in README as synthetic/illustrative or generate them from real runs.
- Update `SUBMISSION.md` to reflect fixed paths and objective direction.
- Update `AGENTS.md` to remove fixed known issues and add new conventions (testing, CI).

### 4.3 Testing & quality (W3)

- Create `tests/` directory mirroring `src/`:
  - `tests/pinn/test_model.py` — forward pass shapes, MC Dropout returns finite std.
  - `tests/pinn/test_physics.py` — `compute_physics_loss` returns finite scalar losses.
  - `tests/pinn/test_loss_balancer.py` — weights sum to `num_losses`.
  - `tests/test_generate_synthetic_data.py` — valid HDF5 with expected groups.
  - `tests/test_config.py` — YAML loads and contains required keys.
- Add `pytest` configuration in `pyproject.toml`.
- Add GitHub Actions workflow that installs dependencies and runs `pytest` on Python 3.10/3.11.
- Add `.pre-commit-config.yaml` with `ruff` (lint + format) and `trailing-whitespace` hooks.

### 4.4 Developer experience (W4)

- Add `pyproject.toml`:
  - `[project]` metadata (name, version, description, authors, license).
  - `[project.optional-dependencies]` for `dev` and `notebooks`.
  - `[tool.pytest.ini_options]` and `[tool.ruff]` configuration.
- Add `environment.yml` for Conda users.
- Add `LICENSE` (MIT, matching README badge).
- Add `CONTRIBUTING.md` with setup, test, and PR instructions.
- Add `CHANGELOG.md` with sections for unreleased changes.
- Add `.github/ISSUE_TEMPLATE/bug_report.md` and `.github/ISSUE_TEMPLATE/feature_request.md`.
- Add `.github/PULL_REQUEST_TEMPLATE.md`.

### 4.5 Research & planning (W5)

- Create `docs/adr/` with at least:
  - `0001-pinn-output-choice.md` — why the model predicts quality metrics instead of temperature/stress fields.
  - `0002-physics-loss-from-quality-metrics.md` — how physics residuals are derived from quality-metric outputs.
  - `0003-multi-objective-optimizer-choice.md` — why NSGA-III is the default and Bayesian opt is secondary.
  - `0004-synthetic-data-strategy.md` — role of synthetic data vs. FEA vs. experiments.
- Create `docs/research/literature-survey.md` summarizing key references for LPBF PINNs, UQ, and multi-objective optimization.
- Update `todo.md` to reflect completed Phase 1 items and redefine remaining phases with clearer milestones.

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| `physics.py` rewrite changes loss landscape and training may diverge. | Keep a fallback flag to disable physics loss; validate on synthetic data first; add tests. |
| Ax/BoTorch API drift. | Pin versions in `requirements.txt` and add a minimal integration test. |
| macOS OpenMP / multiprocessing issues. | Make `num_workers` configurable and default to 0 on macOS. |
| Large HDF5 artifacts in CI. | Generate tiny synthetic datasets in tests; do not commit `data/processed/`. |
| User expectations of a fully validated LPBF digital twin. | Clearly document stub status of validation modules in README and code. |

## 6. Definition of done

- [ ] `pytest` passes locally and in CI.
- [ ] `python src/generate_synthetic_data.py --config data/params.yaml` succeeds.
- [ ] `python src/pinn/train.py --config data/params.yaml` completes without errors.
- [ ] `python src/optimiser/nsga3.py --config data/params.yaml --model data/models/latest/checkpoints/best_model.pt` runs and respects objective directions.
- [ ] All hardcoded paths removed from `data/params.yaml`.
- [ ] `requirements.txt` installs cleanly in a fresh venv.
- [ ] README Quick Start works from a clean clone.
- [ ] `data/models/latest` symlink is created after training.
- [ ] LICENSE, CONTRIBUTING, CHANGELOG, issue templates present.
- [ ] ADRs and research survey committed.

## 7. Open questions

1. Should we keep the current single-headed PINN or add a separate temperature/stress output head? (Design currently keeps single head.)
2. Should the default epoch count be 50, 100, or left at 2 for CI? (Design proposes 50 with CI override.)
3. Should validation stubs be removed entirely or kept with warnings? (Design proposes keep with warnings.)
4. Should the repo create a `data/models/latest` symlink after training for easier optimizer invocation? (Design proposes adding it.)
