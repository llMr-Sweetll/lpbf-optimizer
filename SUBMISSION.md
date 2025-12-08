# LPBF Optimizer - Submission Instructions

This repository contains the working implementation of the LPBF Optimizer.

## Prerequisites

- Python 3.11+
- Conda environment (recommended)
- Dependencies installed via `pip install -r requirements.txt` (ensure `pymoo` is installed)

## How to Run

### 1. Data Generation

Generate synthetic data for training.

```bash
python src/generate_synthetic_data.py --config data/params.yaml --scan-vectors 10 --points-per-vector 100
```

### 2. Model Training

Train the PINN model.

```bash
python src/pinn/train.py --config data/params.yaml
```

*Note: Configured for 2 epochs for testing. Edit `data/params.yaml` implementation for full training.*

### 3. Optimization

Run the multi-objective optimization using the trained model.
Replace the timestamp in the model path with the actual one generated in step 2.

```bash
python src/optimiser/nsga3.py --config data/params.yaml --model "data/models/YOUR_TIMESTAMP/checkpoints/best_model.pt"
```

## Output Locations

- Data: `data/processed/lpbf_dataset.h5`
- Models: `data/models/`
- Optimization Results: `data/optimized/`
