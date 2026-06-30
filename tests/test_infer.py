"""Tests for real-time inference CLI and API."""

import json

import pytest
import torch
import yaml

from infer import build_input, load_model, predict
from pinn.model import PINN


@pytest.fixture
def tiny_config(tmp_path):
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
            "param_bounds": {
                "P": [150.0, 400.0],
                "v": [500.0, 1500.0],
            },
            "objectives": ["residual_stress", "porosity", "geometric_accuracy"],
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def tiny_checkpoint(tiny_config, tmp_path):
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


def test_load_model(tiny_config, tiny_checkpoint):
    with open(tiny_config, "r") as f:
        config = yaml.safe_load(f)
    model, device = load_model(config, tiny_checkpoint)
    assert isinstance(model, PINN)
    assert not model.training


def test_build_input():
    x = build_input([200.0, 800.0], 6, torch.device('cpu'))
    assert x.shape == (1, 6)
    assert x[0, 0].item() == pytest.approx(200.0)
    assert x[0, 1].item() == pytest.approx(800.0)


def test_predict(tiny_config, tiny_checkpoint):
    with open(tiny_config, "r") as f:
        config = yaml.safe_load(f)
    result = predict(config, tiny_checkpoint, [200.0, 800.0])
    assert "prediction" in result
    assert set(result["prediction"].keys()) == set(config["optimizer"]["objectives"])


def test_predict_mc_dropout(tiny_config, tiny_checkpoint):
    with open(tiny_config, "r") as f:
        config = yaml.safe_load(f)
    result = predict(config, tiny_checkpoint, [200.0, 800.0], mc_dropout=True, num_samples=10)
    assert "prediction" in result
    assert "uncertainty" in result
    assert all(v >= 0.0 for v in result["uncertainty"].values())


def test_infer_cli(tiny_config, tiny_checkpoint):
    import os
    import subprocess
    import sys

    params = json.dumps({"P": 250.0, "v": 900.0})
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "src/infer.py", "--config", str(tiny_config),
         "--model", tiny_checkpoint, "--params", params],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    output = json.loads(result.stdout)
    assert "prediction" in output


def test_api_endpoint(tiny_config, tiny_checkpoint):
    fastapi = pytest.importorskip("fastapi")
    from api import app

    client = fastapi.testclient.TestClient(app)
    response = client.post(
        "/predict",
        json={
            "config_path": str(tiny_config),
            "model_path": tiny_checkpoint,
            "params": {"P": 250.0, "v": 900.0},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
