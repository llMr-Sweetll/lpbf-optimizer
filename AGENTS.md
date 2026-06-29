# LPBF-Optimizer — Agent Guide

> This file is written for AI coding agents. It describes the project as it actually exists today, based on the source files, documentation, and configuration in the repository.

## Project overview

**LPBF-Optimizer** is a Python research prototype that combines a Physics-Informed Neural Network (PINN) with multi-objective optimization for Laser Powder Bed Fusion (LPBF) additive manufacturing.

The stated goal is to predict LPBF part quality metrics from process parameters and then find Pareto-optimal process settings. The three quality metrics used are:

- Residual stress
- Porosity
- Geometric accuracy

The project is at the "research prototype" stage. Validation modules are stubs, and the default training data is physics-inspired rather than experimentally validated.

## Technology stack

- **Language:** Python 3.10 or 3.11 (recommended)
- **Deep learning:** PyTorch 2.0+
- **Multi-objective optimization:** pymoo (NSGA-III)
- **Bayesian optimization:** Ax / BoTorch
- **ODE solving:** torchdiffeq
- **Data storage:** HDF5 via h5py
- **Visualization:** matplotlib, seaborn, tensorboard
- **Image/mesh processing:** scikit-image, numpy-stl
- **Configuration:** YAML (`data/params.yaml`)
- **Testing:** pytest with a `tests/` suite mirroring `src/`
- **Linting:** ruff
- **CI:** GitHub Actions (`.github/workflows/ci.yml`)

Dependencies are declared in `requirements.txt`, `pyproject.toml`, and `environment.yml`.

## Project structure

```text
.
├── data/
│   ├── models/                 # Trained model checkpoints (timestamped subdirectories + `latest` symlink)
│   ├── processed/              # Processed HDF5 datasets
│   └── params.yaml             # Central configuration file
├── docs/                       # Markdown documentation, diagrams, ADRs, and GIFs
├── notebooks/                  # Jupyter notebooks with workflow examples
├── src/
│   ├── fea_runner.py           # Wrapper for ABAQUS/COMSOL FEA parameter sweeps
│   ├── generate_synthetic_data.py  # Generate synthetic training data
│   ├── preprocessing.py        # Convert raw FEA outputs into HDF5 training sets
│   ├── optimiser/
│   │   ├── bayesopt.py         # Bayesian optimization (Ax/BoTorch)
│   │   └── nsga3.py            # NSGA-III multi-objective optimization
│   ├── pinn/
│   │   ├── model.py            # PINN architecture with MC Dropout
│   │   ├── physics.py          # Physics-informed residuals
│   │   ├── loss_balancer.py    # GradNorm-style adaptive loss weighting
│   │   └── train.py            # PINN training loop
│   ├── validate/
│   │   ├── build_runner.py     # Convert parameters to G-code / machine formats
│   │   └── characterise.py     # XCT/EBSD/stress characterization and validation
│   └── vis/
│       ├── animate_optimization.py
│       ├── animate_training.py
│       └── animate_training_metrics.py
├── tests/                      # pytest suite
├── README.md
├── requirements.txt
├── pyproject.toml
├── environment.yml
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── SUBMISSION.md               # Quick run instructions
└── todo.md                     # Development roadmap
```

## Runtime architecture

The intended workflow is:

1. **Obtain training data**
   - Option A: `src/generate_synthetic_data.py` creates a synthetic dataset from physics-inspired empirical equations.
   - Option B: `src/fea_runner.py` runs ABAQUS or COMSOL parameter sweeps.
   - Option C: `src/preprocessing.py` converts existing raw FEA `.h5` files into a training set.

2. **Train the PINN surrogate**
   - `src/pinn/train.py` loads the HDF5 dataset, trains `PINN`, and saves checkpoints and plots under `data/models/<timestamp>/`.
   - A `data/models/latest` symlink points to the most recent run for stable downstream commands.

3. **Optimize process parameters**
   - `src/optimiser/nsga3.py` loads the trained checkpoint and runs NSGA-III to find Pareto-optimal parameters.
   - Alternatively, `src/optimiser/bayesopt.py` runs single-objective Bayesian optimization.

4. **Validate experimentally (optional / stub)**
   - `src/validate/build_runner.py` generates coupon STL/G-code/machine job files.
   - `src/validate/characterise.py` analyzes XCT, EBSD, and stress data and compares them to predictions.

5. **Visualize**
   - `src/vis/` contains standalone scripts that generate demonstration GIFs saved to `docs/`.

### Input / output dimensions

The model expects inputs of dimension `model.input_dim` (default 10), formed by concatenating:

- Process parameters (6 in the default config): `P`, `v`, `h`, `theta`, `l_island`, `layer_thickness`
- Spatial coordinates (3): `x`, `y`, `z`
- Time (1)

The model outputs three values:

- Residual stress
- Porosity
- Geometric accuracy

## Build and run commands

There is no build step. Install dependencies and run scripts directly.

### Installation

```bash
# Recommended: create a virtual environment first
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Standard workflow

```bash
# 1. Generate synthetic data
python src/generate_synthetic_data.py --config data/params.yaml --scan-vectors 50 --points-per-vector 64

# 2. Train the PINN
python src/pinn/train.py --config data/params.yaml

# 3. Run NSGA-III optimization using the latest checkpoint
python src/optimiser/nsga3.py \
    --config data/params.yaml \
    --model data/models/latest/checkpoints/best_model.pt
```

### Other useful commands

```bash
# Preprocess real FEA outputs
python src/preprocessing.py --config data/params.yaml [--torch]

# Run an FEA parameter sweep (requires ABAQUS or COMSOL installation)
python src/fea_runner.py --config data/params.yaml --params <scan_vectors.yaml> [--parallel]

# Bayesian optimization
python src/optimiser/bayesopt.py --config data/params.yaml --model <checkpoint.pt> [--multi]

# Generate visualization GIFs
python src/vis/animate_training.py
python src/vis/animate_optimization.py
python src/vis/animate_training_metrics.py

# Generate a build file for a machine
python src/validate/build_runner.py --config data/params.yaml --params <pareto_solutions.csv>

# Characterize experimental samples
python src/validate/characterise.py --config data/params.yaml --xct <xct.h5> --ebsd <ebsd.ang> --stress <stress.csv>
```

## Configuration

`data/params.yaml` is the single source of truth for material properties, model architecture, training hyperparameters, FEA settings, and optimizer settings.

Important sections:

- `material_properties` — Ti-6Al-4V defaults (density, conductivity, absorption, etc.)
- `model` — `input_dim`, `output_dim`, hidden layer width/depth, dropout
- `data` — paths to raw/processed data and train/val/test split ratios
- `training` — epochs, batch size, optimizer, scheduler, loss weights, checkpoint/plot frequency
- `fea` — solver type, executable paths, template path, CPU count
- `optimizer` — algorithm (`nsga3` or `bayesopt`), parameter bounds, objectives, population size, generations

## Code style and conventions

### Language

All source code, docstrings, and documentation are written in **English**.

### Formatting

- 4-space indentation.
- PEP 8-ish naming: `snake_case` for functions and variables, `PascalCase` for classes, `UPPERCASE` for some constants.
- Classes and public methods include verbose Google-style docstrings with `Args:` and `Returns:` sections.
- Inline comments explain physics or workarounds.

### Import style

Modules use standard library imports first, then third-party, then local.

### File organization

- Each major module is a single file with a main orchestrator class and a `main()` CLI entry point.
- Scripts are designed to be run as `python src/<module>/<file>.py --config data/params.yaml ...`.
- Output files are written to timestamped subdirectories under `data/models/`, `data/optimized/`, or `builds/`.

### Logging

Most modules set up `logging.basicConfig` at module load time with a consistent format:

```text
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## Testing instructions

Tests live in the top-level `tests/` directory and mirror the `src/` layout.

Run the full suite from the repository root:

```bash
pytest
```

Key existing tests include:

- `PINN` forward pass with expected input/output shapes.
- `compute_physics_loss` returns finite scalar losses.
- `AdaptiveLossBalancer.update_weights` returns weights that sum to `num_losses`.
- `SyntheticDataGenerator.generate()` produces a valid HDF5 file with the expected groups.
- YAML configuration loads successfully and contains required keys.

CI runs `pytest` on Python 3.10 and 3.11 and lints with `ruff`.

## Security considerations

- The project does not handle network requests, authentication, secrets, or sensitive data by default.
- `src/validate/build_runner.py` contains a `send_to_machine()` method that can perform a dry-run "send" to a configured machine IP. The default is `dry_run=True`; do not set `dry_run=False` unless you intend to interface with real hardware.
- `src/fea_runner.py` shells out to configured external executables (`abaqus`, `comsol`, `javac`, `java`). Ensure these paths come from trusted configuration and are not user-controlled.
- Generated machine files (G-code, `.eosjob`, `.cls`) are written to disk; validate contents before sending them to production LPBF equipment.
- HDF5 files can be large; ensure sufficient disk space when generating datasets or running optimizations.

## Known issues and caveats

The following items are genuinely outstanding and should be reviewed before making changes:

1. **Validation modules are stubs:** `build_runner.py` and `characterise.py` contain placeholder logic and dummy data fallbacks for missing machine interfaces or file parsers.
2. **FEA runner requires external licenses:** `src/fea_runner.py` shells out to ABAQUS or COMSOL, which must be installed and licensed separately. The default `fea.comsol_java_path` is left empty and must be configured locally before use.
3. **Default data is synthetic:** `src/generate_synthetic_data.py` produces physics-inspired empirical data for quick smoke tests. It is not a substitute for validated FEA or experimental ground truth.
4. **OpenMP conflict workaround:** `KMP_DUPLICATE_LIB_OK=TRUE` is set inside training and optimization scripts.
5. **Input dimension handling** in `train.py` trims `S_batch` if the concatenated input exceeds `model.input_dim`. This is robust to the default 10-dimensional input but should be revisited if parameter order or coordinate dimensions change.

## Development roadmap

See `todo.md` for the full roadmap. Major planned phases include:

- Phase 1: Robustness, experimental validation, uncertainty quantification
- Phase 2: Melt pool CFD and grain-structure modeling
- Phase 3: Real-time data assimilation and closed-loop control
- Phase 4: Scan path optimization and functionally graded materials
- Phase 5: Full 3D simulation and microstructure evolution

## Useful references embedded in the code

- Gal & Ghahramani (2016) — MC Dropout
- Wang, Teng & Perdikaris (2021) — Gradient pathology / GradNorm
- Deb & Jain (2014) — NSGA-III
