# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- Physics loss module aligned with PINN quality-metric outputs.
- NSGA-III now maximises geometric accuracy internally.
- Removed hardcoded Windows paths from configuration.
- Added missing `seaborn` and `scikit-learn` dependencies.

### Added
- Initial test suite under `tests/`.
- GitHub Actions CI workflow.
- `pyproject.toml`, `environment.yml`, `LICENSE`, `CONTRIBUTING.md`.
- Architecture Decision Records (ADRs) and research literature survey.

### Changed
- README Quick Start now includes data generation step.
- `data/params.yaml` uses relative paths and new training fields.
