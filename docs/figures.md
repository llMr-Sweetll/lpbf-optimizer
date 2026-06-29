# Figures and Visualisations

This directory contains generated figures and animations used by the documentation.

| File | Source | Description |
|------|--------|-------------|
| `melt_pool_dynamics.gif` | `src/vis/animate_training.py` | Illustrative analytic temperature field (not PINN inference). |
| `optimization_evolution.gif` | `src/vis/animate_optimization.py` | Synthetic NSGA-III evolution illustration. |
| `loss_history.gif` | `src/vis/animate_training_metrics.py` | Synthetic loss-curve illustration. |
| `weights_evolution.gif` | `src/vis/animate_training_metrics.py` | Synthetic weight-evolution illustration. |
| `loss_curves.png` | `src/pinn/train.py` | Real training/validation loss curves. |
| `loss_components.png` | `src/pinn/train.py` | Data vs physics loss components. |
| `adaptive_weights.png` | `src/pinn/train.py` | Adaptive loss weights from GradNorm. |
| `pareto_front_3d.png` | `src/optimiser/nsga3.py` | Pareto front from NSGA-III. |
