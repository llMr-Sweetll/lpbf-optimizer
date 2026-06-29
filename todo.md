# LPBF-Optimizer Roadmap

This roadmap links high-level phases to GitHub issues. Issue numbers are current as of 2026-06-30.

## Status Overview

- **Latest release:** [v0.2.0](https://github.com/llMr-Sweetll/lpbf-optimizer/releases/tag/v0.2.0)
- **Health checks:** `pytest` 17/17 passed, `ruff` clean, `python -m build` succeeds
- **Open issues:** tracked at https://github.com/llMr-Sweetll/lpbf-optimizer/issues

## Recently Completed in v0.2.0

- [x] [#2](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/2) Fix synthetic data train/val/test leakage
- [x] [#4](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/4) Enforce physical bounds on PINN outputs
- [x] [#6](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/6) Expand literature survey: LPBF PINNs 2024–2025
- [x] [#7](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/7) Expand literature survey: uncertainty quantification beyond MC Dropout
- [x] [#8](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/8) Expand literature survey: multi-objective Bayesian optimisation
- [x] [#9](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/9) Expand literature survey: in-situ sensors and digital twins for LPBF
- [x] [#10](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/10) Expand literature survey: reduced-order CFD and grain-structure models
- [x] [#11](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/11) Add input normalisation to training pipeline
- [x] [#13](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/13) Implement physics ablation study script
- [x] [#20](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/20) Prepare v0.2.0 release notes and git tag
- [x] [#21](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/21) Publish Docker image for reproducible environment
- [x] [#23](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/23) Set up automated release workflow

## Phase 1: Robustness & Scientific Rigor (target v0.2.x patch releases)

- [x] [#2](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/2) Fix synthetic data train/val/test leakage
- [x] [#11](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/11) Add input normalisation to training pipeline
- [x] [#4](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/4) Enforce physical bounds on PINN outputs
- [x] [#13](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/13) Implement physics ablation study script
- [ ] [#12](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/12) Add optional temperature/stress output head for rigorous PDE enforcement
- [ ] [#5](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/5) Review `predict_with_uncertainty` for future architecture changes

## Phase 2: Validation & Real Data (target v0.3.0)

- [ ] [#18](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/18) Flesh out validation module with XCT/EBSD/stress comparison
- [ ] Calibrate synthetic generator against real FEA or literature benchmarks
- [ ] [#16](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/16) Add data versioning and lineage tracking

## Phase 3: Advanced Optimisation (target v0.4.0)

- [ ] [#3](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/3) Migrate Bayesian optimisation from deprecated `ax.service.managed_loop.optimize` to `ax.api.client`
- [ ] [#17](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/17) Add robust / reliability-based optimisation under MC Dropout uncertainty
- [ ] [#15](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/15) Add hyperparameter search with Optuna

## Phase 4: CFD, Sensors, and Digital Twin (target v0.5.0)

- [ ] Reduced-order melt-pool surrogate
- [ ] In-situ sensor assimilation stubs
- [ ] Closed-loop control prototype

## Phase 5: Research & Literature Expansion

- [x] [#6](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/6) Expand literature survey: LPBF PINNs 2024–2025
- [x] [#7](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/7) Expand literature survey: uncertainty quantification beyond MC Dropout
- [x] [#8](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/8) Expand literature survey: multi-objective Bayesian optimisation
- [x] [#9](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/9) Expand literature survey: in-situ sensors and digital twins for LPBF
- [x] [#10](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/10) Expand literature survey: reduced-order CFD and grain-structure models

## Phase 6: Release & Distribution

- [x] [#20](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/20) Prepare v0.2.0 release notes and git tag
- [x] [#21](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/21) Publish Docker image for reproducible environment
- [ ] [#22](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/22) Publish `lpbf-optimizer` to PyPI
- [x] [#23](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/23) Set up automated release workflow

## Phase 7: Documentation & Developer Experience

- [ ] [#27](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/27) Add notebook walkthrough for each workflow step
- [ ] [#24](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/24) Add ‘Contributing a new physics residual’ guide
- [ ] [#25](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/25) Add benchmark / regression test results to README
- [ ] [#14](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/14) Add experiment tracking integration (TensorBoard / Weights & Biases)
- [ ] [#26](https://github.com/llMr-Sweetll/lpbf-optimizer/issues/26) Add real-time inference CLI / API
