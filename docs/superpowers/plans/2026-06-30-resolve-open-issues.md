# Resolve Remaining Open Issues — Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or inline execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all currently open GitHub issues (#3, #5, #12, #14, #15, #16, #17, #18, #22, #24, #25, #26, #27) by implementing the acceptance criteria, updating documentation, and verifying with tests.

**Architecture:** Work in topical batches — core PINN/optimisation first, then validation/data/tooling, then documentation/PyPI — so each batch leaves the repo in a consistent, passing state. Independent modules (validation, notebooks, hyperparameter search) can be delegated to focused subagents to run in parallel.

**Tech Stack:** Python 3.10–3.12, PyTorch, Ax 1.3 (`ax.api.client`), pymoo, Optuna, TensorBoard/W&B, scikit-image, pandas, FastAPI (optional).

---

## Batch 1: Core PINN & Uncertainty

### Task 1.1: #5 — Stabilise `predict_with_uncertainty` and add backend interface

**Files:**
- Create: `src/pinn/uncertainty.py`
- Modify: `src/pinn/model.py:232-269`
- Test: `tests/pinn/test_uncertainty.py`

- [ ] **Step 1: Add backend interface**
  Create `src/pinn/uncertainty.py` with `MCDropoutBackend` and `DeepEnsembleBackend` classes. `MCDropoutBackend.predict(model, x, n_samples)` must enable only `nn.Dropout` layers during inference (not whole `train()` mode) to avoid future BatchNorm side-effects, and restore all layer states afterward.
- [ ] **Step 2: Refactor `PINN.predict_with_uncertainty`**
  Replace the manual `self.train()`/`finally` block with `MCDropoutBackend.predict(self, x, num_samples)`. Document assumptions in the docstring.
- [ ] **Step 3: Write tests**
  - Mode restoration test: call `predict_with_uncertainty` while model is in `.eval()`, assert it returns to `.eval()`.
  - Output shape test: `(mean, std)` shapes match input batch/output dims.
  - Backend registry test: `get_uncertainty_backend('mc_dropout')` returns the right class.
- [ ] **Step 4: Run tests**
  `pytest tests/pinn/test_uncertainty.py -v` → PASS
- [ ] **Step 5: Commit**
  `git add src/pinn/uncertainty.py src/pinn/model.py tests/pinn/test_uncertainty.py && git commit -m "feat(pinn): add pluggable uncertainty backend and stabilise MC Dropout"`

---

### Task 1.2: #12 — Optional trainable temperature output head

**Files:**
- Modify: `src/pinn/model.py`, `src/pinn/physics.py`, `src/pinn/train.py`, `data/params.yaml`
- Test: `tests/pinn/test_temperature_head.py`

- [ ] **Step 1: Add `predict_temperature` config and model head**
  Add `model.predict_temperature: false` to `data/params.yaml`. In `PINN.__init__`, when enabled, build an additional `temperature_head` Linear layer mapping last hidden width to 1 and set `self.has_temperature_head = True`.
- [ ] **Step 2: Forward pass exposes temperature when requested**
  Add `PINN.forward(..., return_temperature=False)`. When `self.has_temperature_head` and `return_temperature=True`, return `(quality_outputs, temperature)`. Default returns only quality outputs for backward compatibility.
- [ ] **Step 3: Heat residual can use predicted temperature**
  Extend `compute_physics_loss` signature with `use_predicted_temperature=False`. When true, take predicted `T` from model output (requires `requires_grad` coords/t) and pass it to `_heat_residual`; otherwise keep analytic field.
- [ ] **Step 4: Trainer computes temperature supervision**
  In `PINNTrainer.train_step`, if `predict_temperature` is enabled, compute analytic temperature target with `requires_grad=False`, add MSE loss on the temperature head, and add a `temperature` key to the returned losses dict.
- [ ] **Step 5: Ablation comparison**
  Add a test that trains one step with `predict_temperature=true` and one with `false`, asserting both converge (loss decreases) and that the predicted-temperature path produces a non-zero temperature gradient.
- [ ] **Step 6: Commit**
  `git add src/pinn/model.py src/pinn/physics.py src/pinn/train.py data/params.yaml tests/pinn/test_temperature_head.py && git commit -m "feat(pinn): optional trainable temperature output head"`

---

### Task 1.3: #17 — Robust NSGA-III under MC Dropout uncertainty

**Files:**
- Modify: `src/optimiser/nsga3.py`, `data/params.yaml`
- Test: `tests/optimiser/test_nsga3_robust.py`

- [ ] **Step 1: Add robust objective transform**
  Add `beta: 2.0` under `optimizer` config. In `SurrogateProblem._evaluate`, when `self.robust=True`, call `model.predict_with_uncertainty` and compute objectives as `mean + beta * std` for minimised objectives; for `geometric_accuracy` use `mean - beta * std` (negated).
- [ ] **Step 2: CLI flag `--robust`**
  Add `--robust` to `nsga3.py` main and pass it to `SurrogateProblem`.
- [ ] **Step 3: Test robust evaluation**
  Unit-test that enabling `--robust` changes objective values compared to deterministic mode for a model with non-zero dropout.
- [ ] **Step 4: Commit**
  `git add src/optimiser/nsga3.py data/params.yaml tests/optimiser/test_nsga3_robust.py && git commit -m "feat(optimiser): robust NSGA-III with MC Dropout uncertainty"`

---

## Batch 2: Optimisation & Inference

### Task 2.1: #3 — Migrate Bayesian optimisation to `ax.api.client`

**Files:**
- Modify: `src/optimiser/bayesopt.py`
- Test: `tests/optimiser/test_bayesopt.py`

- [ ] **Step 1: Replace `optimize` with `Client`**
  Use `from ax.api.client import Client` and `from ax.api.configs import RangeParameterConfig`. Build parameters as `RangeParameterConfig(name, bounds, parameter_type='float')`. Configure experiment, configure optimization (`objective='objective'` or `'-'` prefix for maximization), run trials with `get_next_trials` / `complete_trial`, then `get_best_parameterization`.
- [ ] **Step 2: Preserve `--multi` sequential behaviour**
  Keep `run_multi_objective_optimization` iterating over `objectives` and calling the single-objective routine.
- [ ] **Step 3: Test single-objective run**
  Mock the model forward and run 3 trials; assert a CSV and HDF5 are produced and `best_parameters` is returned.
- [ ] **Step 4: Commit**
  `git add src/optimiser/bayesopt.py tests/optimiser/test_bayesopt.py && git commit -m "refactor(optimiser): migrate Bayesian optimisation to ax.api.client"`

---

### Task 2.2: #26 — Real-time inference CLI / API

**Files:**
- Create: `src/infer.py`, `src/api.py`
- Modify: `pyproject.toml`, `README.md`
- Test: `tests/test_infer.py`

- [ ] **Step 1: Create inference CLI**
  `src/infer.py` loads a checkpoint and config, builds `PINN`, parses `--params '{"P":300,...}'` (JSON), appends dummy coords/time, prints predictions. Support `--mc-dropout` for uncertainty.
- [ ] **Step 2: Optional FastAPI endpoint**
  `src/api.py` exposes `/predict` and `/predict/uncertain` endpoints using the same loader.
- [ ] **Step 3: Console entry point**
  Add `[project.scripts]` `lpbf-optimizer-infer = "infer:main"` in `pyproject.toml`.
- [ ] **Step 4: Test CLI and API**
  Test that CLI returns expected shape; test API client with `TestClient` if FastAPI installed.
- [ ] **Step 5: Commit**
  `git add src/infer.py src/api.py pyproject.toml tests/test_infer.py && git commit -m "feat(cli): add real-time inference CLI and API"`

---

## Batch 3: Training Infrastructure

### Task 3.1: #14 — Experiment tracking integration

**Files:**
- Create: `src/pinn/trackers.py`
- Modify: `src/pinn/train.py`, `data/params.yaml`, `pyproject.toml`
- Test: `tests/pinn/test_trackers.py`

- [ ] **Step 1: Add tracker abstraction**
  `BaseTracker`, `TensorBoardTracker`, `WandbTracker`, `NoopTracker`. Optional `wandb` import guarded.
- [ ] **Step 2: Wire into trainer**
  Read `training.tracker` config (`tensorboard` / `wandb` / `none`). Log losses, learning rate, and adaptive weights each epoch; log generated plots at end of training.
- [ ] **Step 3: Optional dependencies**
  Add `wandb` to `project.optional-dependencies.dev`? Keep it optional and document.
- [ ] **Step 4: Test**
  Assert TensorBoard event files are created when configured; assert NoopTracker does nothing.
- [ ] **Step 5: Commit**
  `git add src/pinn/trackers.py src/pinn/train.py data/params.yaml pyproject.toml tests/pinn/test_trackers.py && git commit -m "feat(train): add optional TensorBoard/W&B experiment tracking"`

---

### Task 3.2: #16 — Data versioning and lineage

**Files:**
- Create: `src/data/lineage.py`
- Modify: `src/generate_synthetic_data.py`, `src/pinn/train.py`, `src/optimiser/nsga3.py`, `src/optimiser/bayesopt.py`, `data/params.yaml`
- Test: `tests/test_lineage.py`

- [ ] **Step 1: Lightweight manifest utilities**
  `compute_file_hash(path)`, `write_manifest(config_path, dataset_path, output_dir)`, producing YAML with config hash, dataset hash, timestamp, git commit (if available).
- [ ] **Step 2: Embed lineage in artifacts**
  Save manifest next to processed HDF5 and include `lineage` dict in checkpoints, optimisation results, and BayesOpt outputs.
- [ ] **Step 3: Config flag**
  Add `data.lineage_manifest: true` to `data/params.yaml`.
- [ ] **Step 4: Test**
  Generate data, verify manifest exists and hash matches; verify checkpoint contains lineage.
- [ ] **Step 5: Commit**
  `git add src/data/lineage.py src/generate_synthetic_data.py src/pinn/train.py src/optimiser/nsga3.py src/optimiser/bayesopt.py data/params.yaml tests/test_lineage.py && git commit -m "feat(data): add lightweight dataset lineage manifest"`

---

### Task 3.3: #15 — Hyperparameter search with Optuna

**Files:**
- Create: `scripts/hyperparameter_search.py`, `tests/test_hyperparameter_search.py`
- Modify: `pyproject.toml` (optional `optuna`), `README.md`

- [ ] **Step 1: Objective function**
  Load a small fixed dataset, build `PINNTrainer` with trial-sampled width/depth/lr/lambda weights, run a reduced number of epochs, return validation loss.
- [ ] **Step 2: Optuna study**
  Create `optuna.create_study(direction='minimize')`, run `n_trials`, prune with `optuna.TrialPruned` if val loss stalls.
- [ ] **Step 3: Output**
  Save `best_config.yaml`, `study.db`, and `convergence.png`.
- [ ] **Step 4: Test**
  Run 2 trials in test mode and assert outputs are created.
- [ ] **Step 5: Commit**
  `git add scripts/hyperparameter_search.py tests/test_hyperparameter_search.py pyproject.toml README.md && git commit -m "feat(hpo): add Optuna hyperparameter search script"`

---

## Batch 4: Validation Module

### Task 4.1: #18 — Flesh out validation parsers and report

**Files:**
- Modify: `src/validate/characterise.py`
- Create: `src/validate/report.py`, `tests/validate/test_characterise.py`, sample data under `tests/data/`

- [ ] **Step 1: Robust CSV parsers**
  - XCT porosity CSV: required column `porosity`; optional `x,y,z`.
  - EBSD CSV: required `phi1,phi,phi2,x,y`; optional `ci,phase`.
  - Stress CSV: required `sigma_xx`; optional `sigma_yy,sigma_zz,x,y,z`.
  Return parsed DataFrames; fail with clear errors instead of dummy data.
- [ ] **Step 2: Comparison metrics**
  `compare_with_predictions` computes MAE/RMSE per quantity and overall, plus percent error; outputs a JSON/YAML report.
- [ ] **Step 3: Report generator**
  `src/validate/report.py` creates a Markdown report with tables and figures.
- [ ] **Step 4: Tests with sample CSVs**
  Create minimal valid CSVs in `tests/data/`, test parsing, metrics, and report generation.
- [ ] **Step 5: Commit**
  `git add src/validate/characterise.py src/validate/report.py tests/validate/test_characterise.py tests/data/ && git commit -m "feat(validate): robust XCT/EBSD/stress parsers and validation report"`

*This batch can be delegated to a focused subagent working only in `src/validate/` and `tests/validate/`.*

---

## Batch 5: Packaging & Documentation

### Task 5.1: #22 — Prepare PyPI publishing

**Files:**
- Modify: `pyproject.toml`, `README.md`
- Create: `MANIFEST.in`

- [ ] **Step 1: Console scripts**
  Add `[project.scripts]` for `lpbf-optimizer-train`, `lpbf-optimizer-infer`, `lpbf-optimizer-generate`, `lpbf-optimizer-optimise`, `lpbf-optimizer-characterise`.
- [ ] **Step 2: MANIFEST.in**
  Include `LICENSE`, `README.md`, `data/params.yaml`, docs, and notebooks in sdist.
- [ ] **Step 3: PyPI badge**
  Add `https://img.shields.io/pypi/v/lpbf-optimizer` badge to README (will be active after upload).
- [ ] **Step 4: Build and check**
  `python -m build && twine check dist/*` (install `twine` if needed).
- [ ] **Step 5: Commit**
  `git add pyproject.toml MANIFEST.in README.md && git commit -m "chore(packaging): add entry points and MANIFEST.in for PyPI"`

*Note: actual PyPI upload is blocked only by the `PYPI_API_TOKEN` repository secret; leave a comment on #22 explaining this.*

---

### Task 5.2: #24 — Contributing a new physics residual guide

**Files:**
- Create: `docs/contributing-physics-residual.md`
- Modify: `README.md` (link), `docs/adr/physics-residual-extension.md` (optional)

- [ ] **Step 1: Write guide**
  Explain `src/pinn/physics.py` structure, how to add a residual function, wire it into `compute_physics_loss`, register a lambda weight, and add a test.
- [ ] **Step 2: Example residual**
  Include a complete copy-pasteable example (e.g., a simplified surface-roughness residual).
- [ ] **Step 3: Testing checklist**
  List unit tests, shape checks, and finite-difference sanity checks.
- [ ] **Step 4: Commit**
  `git add docs/contributing-physics-residual.md README.md && git commit -m "docs: add guide for contributing a new physics residual"`

---

### Task 5.3: #25 — Benchmark / regression results in README

**Files:**
- Create: `scripts/benchmark.py`, `docs/benchmark_results.md`
- Modify: `README.md`

- [ ] **Step 1: Benchmark script**
  `scripts/benchmark.py` runs generate → train → optimise with default config, times each step, captures final losses, and writes a YAML/Markdown report.
- [ ] **Step 2: Run benchmark**
  Execute with reduced epochs if needed to keep runtime reasonable; capture results.
- [ ] **Step 3: Add table to README**
  Include dataset size, epochs, final train/val losses, physics residual, runtime, hardware.
- [ ] **Step 4: Test**
  Ensure `scripts/benchmark.py --dry-run` works in CI.
- [ ] **Step 5: Commit**
  `git add scripts/benchmark.py docs/benchmark_results.md README.md && git commit -m "docs: add benchmark script and regression results table"`

---

### Task 5.4: #27 — Notebook walkthroughs per workflow step

**Files:**
- Create: `notebooks/01_data_generation.ipynb`, `notebooks/02_training.ipynb`, `notebooks/03_optimisation.ipynb`, `notebooks/04_visualisation.ipynb`
- Modify: `README.md`

- [ ] **Step 1: Split existing notebooks**
  Each notebook focuses on one step, runs end-to-end, and links to the next.
- [ ] **Step 2: Lint notebooks**
  `ruff check notebooks --fix` clean.
- [ ] **Step 3: Update README links**
  Add notebook section in README.
- [ ] **Step 4: Commit**
  `git add notebooks/ README.md && git commit -m "docs(notebooks): add per-step notebook walkthroughs"`

*This batch can be delegated to a documentation-focused subagent.*

---

## Final Integration

- [ ] Run full verification: `ruff check .`, `pytest -q`, `python -m build`
- [ ] Update `todo.md` to mark all completed issues
- [ ] Close all resolved GitHub issues with comments linking to commits/releases
- [ ] Push final `master`
