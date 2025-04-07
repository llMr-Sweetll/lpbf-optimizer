# LPBF PINN Training Metrics Documentation

## Tracked Metrics

### 1. Training Loss (`train_loss`)
- **Definition**: Total loss (data + physics) on training data
- **Range**: Typically starts >1.0, should decrease to <0.1
- **Interpretation**: Lower values indicate better model fitting
- **Convergence Criteria**: <0.05 for 3 consecutive epochs

### 2. Validation Loss (`val_loss`)
- **Definition**: Pure data loss on held-out validation set
- **Significance**: Measures generalization performance
- **Warning Sign**: >20% higher than training loss suggests overfitting

### 3. Data Loss (`data_loss`)
- **Source**: MSE between predicted and FEA-simulated outputs
- **Typical Values**: 0.01-0.5
- **Optimization Target**: Should steadily decrease

### 4. Physics Loss (`physics_loss`)
- **Components**:
  - Heat equation residual
  - Stress equilibrium residual
- **Weighting**: Controlled by `lambda_heat` and `lambda_stress` in config
- **Acceptable Range**: 1e-4 to 1e-2

## Interpretation Guide

### Loss Curves Analysis
- **Healthy Training**: Parallel decreasing train/val curves
- **Overfitting**: Growing gap between train/val losses
- **Underfitting**: High plateaus in both losses

### Numerical Values
- **Absolute Values**:
  - >1.0 - Needs improvement
  - 0.1-1.0 - Moderate performance
  - <0.1 - Good convergence
- **Relative Changes**:
  - <5% change for 10 epochs suggests convergence
  - Sudden spikes indicate instability

## Training Progress Format
- **Epoch [current/total]**: Shows progress through total training cycles
- **Batch [current/total]**: Indicates progress within current epoch
- **Typical Values**:
  - Early Training: High losses (1e4-1e6 range common)
  - Mid Training: 50-80% reduction from initial
  - Converged: <1% of initial loss

## Loss Component Interpretation
### Data Loss
- Measures fit to training data
- Expected range: 1e-1 to 1e4
- Influenced by:
  - Measurement noise levels
  - Model capacity

### Physics Loss
- Enforces physical constraints
- Scaled by lambda parameters (heat: λ=${config.lambda_heat}, stress: λ=${config.lambda_stress})
- Healthy ratio: 0.1-10% of data loss

## Convergence Monitoring
1. **Early Training (First 50 epochs):**
   - Expect 5-15% reduction per epoch
2. **Mid Training (50-200 epochs):**
   - 1-5% reduction per epoch
3. **Final Phase:**
   - <0.5% reduction per epoch

## Troubleshooting High Losses
| Symptom | Possible Causes |
|---------|----------------|
| Both losses high | Insufficient model capacity |
| Physics loss dominant | Increase lambda_heat/stress |
| Data loss oscillates | Reduce learning rate |
| Validation loss rising | Overfitting | Add dropout/regularization |

## Related Files
- Configuration: `data/params.yaml`
- Training Logs: `data/models/TIMESTAMP/logs/`
- Loss Plots: `data/models/TIMESTAMP/plots/`

> **Note**: All values are logged after each epoch in CSV format for analysis