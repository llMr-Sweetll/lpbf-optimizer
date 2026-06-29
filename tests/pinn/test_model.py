import sys
from pathlib import Path

import torch

# Add src/pinn to the path so the standalone modules can be imported.
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src" / "pinn"))

from model import PINN  # noqa: E402


def test_pinn_forward_shape():
    """PINN forward pass should produce the expected input/output shapes."""
    batch_size = 16
    input_dim = 10
    output_dim = 3

    model = PINN(input_dim=input_dim, output_dim=output_dim, width=64, depth=2)
    x = torch.randn(batch_size, input_dim)
    y = model(x)

    assert y.shape == (batch_size, output_dim)


def test_pinn_legacy_arguments_still_work():
    """Legacy ``in_dim`` / ``out_dim`` arguments should remain compatible."""
    model = PINN(in_dim=8, out_dim=2, width=32, depth=2)
    x = torch.randn(4, 8)
    y = model(x)

    assert y.shape == (4, 2)
    assert model.input_dim == 8
    assert model.in_dim == 8


def test_predict_with_uncertainty_preserves_training_mode():
    """MC Dropout inference should leave the model in its original mode."""
    model = PINN(input_dim=10, output_dim=3, width=32, depth=2, dropout_rate=0.1)
    x = torch.randn(8, 10)

    model.train()
    model.predict_with_uncertainty(x, num_samples=5)
    assert model.training is True

    model.eval()
    model.predict_with_uncertainty(x, num_samples=5)
    assert model.training is False


def test_default_input_dimension():
    """Default PINN constructor should match the project config (10 inputs, 3 outputs)."""
    model = PINN()
    assert model.in_dim == 10
    x = torch.rand(4, 10)
    y = model(x)
    assert y.shape == (4, 3)


def test_predict_with_uncertainty_returns_finite_std():
    """MC Dropout should return finite means and non-negative standard deviations."""
    model = PINN(in_dim=10, out_dim=3, dropout_rate=0.1)
    x = torch.rand(8, 10)
    mean, std = model.predict_with_uncertainty(x, num_samples=10)
    assert mean.shape == (8, 3)
    assert std.shape == (8, 3)
    assert torch.all(std >= 0)
    assert torch.all(torch.isfinite(mean))
    assert torch.all(torch.isfinite(std))
