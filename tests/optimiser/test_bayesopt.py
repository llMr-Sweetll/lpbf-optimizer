"""Tests for Bayesian optimisation with the Ax Client API."""

from pathlib import Path

import pytest
import torch
import yaml

from optimiser.bayesopt import BayesianOptimizer
from pinn.model import PINN


@pytest.fixture
def tiny_config(tmp_path):
    """Create a minimal config for fast Bayesian optimisation tests."""
    config = {
        "model": {
            "input_dim": 6,
            "output_dim": 3,
            "hidden_width": 16,
            "hidden_depth": 2,
            "dropout_rate": 0.0,
            "apply_output_bounds": False,
        },
        "optimizer": {
            "output_dir": str(tmp_path / "bayesopt_out"),
            "param_bounds": {
                "P": [150.0, 400.0],
                "v": [500.0, 1500.0],
            },
            "objectives": ["residual_stress", "porosity", "geometric_accuracy"],
            "n_trials": 2,
            "seed": 42,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def tiny_checkpoint(tiny_config, tmp_path):
    """Save a tiny trained-model checkpoint for BayesOpt tests."""
    model = PINN(input_dim=6, output_dim=3, width=16, depth=2, dropout_rate=0.0)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 0,
        "loss": 0.0,
        "metrics": {},
    }
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


def test_single_objective_optimization(tiny_config, tiny_checkpoint):
    optimizer = BayesianOptimizer(str(tiny_config), tiny_checkpoint)
    best_params, best_values, client = optimizer.run_single_objective_optimization(
        objective_idx=0, n_trials=3
    )

    assert isinstance(best_params, dict)
    assert "P" in best_params and "v" in best_params
    assert "objective" in best_values

    output_dir = Path(optimizer.output_dir)
    objective_name = optimizer.config["optimizer"]["objectives"][0]
    assert (output_dir / f"bayesopt_{objective_name}.csv").exists()
    assert (output_dir / f"bayesopt_{objective_name}.h5").exists()
    assert (output_dir / f"bayesopt_convergence_{objective_name}.png").exists()


def test_multi_objective_optimization(tiny_config, tiny_checkpoint):
    optimizer = BayesianOptimizer(str(tiny_config), tiny_checkpoint)
    results = optimizer.run_multi_objective_optimization(n_trials=2)

    assert len(results) == 3
    for best_params, best_values, client in results:
        assert isinstance(best_params, dict)
        assert "P" in best_params
        assert "objective" in best_values
