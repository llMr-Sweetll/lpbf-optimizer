"""Tests for robust NSGA-III optimisation."""


import numpy as np
import pytest
import torch
import yaml

from optimiser.nsga3 import NSGAOptimizer
from pinn.model import PINN


@pytest.fixture
def tiny_config(tmp_path):
    config = {
        "model": {
            "input_dim": 6,
            "output_dim": 3,
            "hidden_width": 16,
            "hidden_depth": 2,
            "dropout_rate": 0.3,
            "apply_output_bounds": False,
        },
        "optimizer": {
            "output_dir": str(tmp_path / "nsga_out"),
            "param_bounds": {
                "P": [150.0, 400.0],
                "v": [500.0, 1500.0],
            },
            "objectives": ["residual_stress", "porosity", "geometric_accuracy"],
            "pop_size": 10,
            "n_gen": 2,
            "n_partitions": 4,
            "beta": 2.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def tiny_checkpoint(tiny_config, tmp_path):
    model = PINN(input_dim=6, output_dim=3, width=16, depth=2, dropout_rate=0.3)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 0,
        "loss": 0.0,
        "metrics": {},
    }
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


def test_robust_changes_objectives(tiny_config, tiny_checkpoint):
    det = NSGAOptimizer(str(tiny_config), tiny_checkpoint, robust=False)
    rob = NSGAOptimizer(str(tiny_config), tiny_checkpoint, robust=True, beta=2.0)

    x = np.array([[200.0, 800.0], [300.0, 1000.0]])
    det_out = {}
    rob_out = {}
    det.problem._evaluate(x, det_out)
    rob.problem._evaluate(x, rob_out)

    assert not np.allclose(det_out["F"], rob_out["F"])
    assert rob_out["F"].shape == (2, 3)


