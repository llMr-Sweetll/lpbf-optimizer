# LPBF-Optimizer — Quick Run

```bash
python src/generate_synthetic_data.py --config data/params.yaml --scan-vectors 50 --points-per-vector 64
python src/pinn/train.py --config data/params.yaml
python src/optimiser/nsga3.py --config data/params.yaml --model data/models/latest/checkpoints/best_model.pt
```

`geometric_accuracy` is maximised by NSGA-III internally (it is negated for pymoo's minimisation convention).
