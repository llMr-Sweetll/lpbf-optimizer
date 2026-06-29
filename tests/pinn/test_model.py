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


def test_pinn_output_bounds_are_applied():
    """When enabled, outputs must lie inside the configured physical ranges."""
    bounds = ((50.0, 800.0), (0.0, 0.3), (0.7, 1.0))
    model = PINN(input_dim=10, output_dim=3, width=32, depth=2,
                 apply_output_bounds=True, output_bounds=bounds)
    x = torch.randn(16, 10) * 10.0  # large inputs to test squashing
    y = model(x)

    assert y.shape == (16, 3)
    for i, (lo, hi) in enumerate(bounds):
        assert torch.all(y[:, i] >= lo - 1e-6)
        assert torch.all(y[:, i] <= hi + 1e-6)


def test_pinn_default_output_bounds():
    """Default bounds should match the standard LPBF quality metric ranges."""
    model = PINN(apply_output_bounds=True, width=32, depth=2)
    bounds = model.get_output_bounds()
    expected = ((50.0, 800.0), (0.0, 0.3), (0.7, 1.0))
    assert len(bounds) == len(expected)
    for (lo, hi), (elo, ehi) in zip(bounds, expected):
        assert abs(lo - elo) < 1e-6
        assert abs(hi - ehi) < 1e-6


def test_pinn_without_bounds_is_unbounded():
    """With bounds disabled, the linear output head should remain unbounded."""
    model = PINN(input_dim=10, output_dim=3, width=32, depth=2,
                 apply_output_bounds=False)
    assert model.get_output_bounds() is None

    # Force the final layer to produce values far outside physical bounds.
    with torch.no_grad():
        model.out.bias.fill_(1e4)

    x = torch.randn(4, 10)
    y = model(x)
    assert torch.any(y > 800.0)
