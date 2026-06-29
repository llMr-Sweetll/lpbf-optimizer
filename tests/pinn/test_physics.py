import pytest
import torch

import sys
from pathlib import Path

# Add src/pinn to the path so the standalone modules can be imported.
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src" / "pinn"))

from model import PINN
from physics import compute_physics_loss


@pytest.fixture
def mat_props():
    return {
        "rho": 4430.0,
        "cp": 526.3,
        "k": 6.7,
        "eta": 0.35,
        "r0": 0.05,
        "Hm": 286000.0,
        "Ts": 1604.0,
        "Tl": 1660.0,
        "E": 110.0,
        "nu": 0.34,
        "alpha": 8.6e-6,
        "sigma_y": 1100.0,
    }


def test_compute_physics_loss_returns_finite_scalar(mat_props):
    """Physics loss should be a finite scalar and all components should be finite."""
    batch_size = 16
    n_params = 6
    model = PINN(input_dim=10, output_dim=3, width=64, depth=2)

    S = torch.rand(batch_size, n_params) * 100.0
    S[:, 2] = S[:, 2] * 0.001 + 0.05  # hatch spacing ~mm
    S[:, 3] = torch.rand(batch_size) * 90.0
    S[:, 4] = torch.rand(batch_size) * 8.0 + 2.0
    S[:, 5] = torch.rand(batch_size) * 0.04 + 0.02  # layer thickness ~mm

    coords = torch.rand(batch_size, 3, requires_grad=True)
    t = torch.rand(batch_size, 1, requires_grad=True)

    heat_loss, stress_loss, porosity_loss, geometry_loss = compute_physics_loss(
        model,
        S,
        coords,
        t,
        mat_props,
        lambda_heat=0.1,
        lambda_stress=0.1,
        lambda_porosity=0.05,
        lambda_geometry=0.05,
        return_components=True,
    )

    total_loss = (
        0.1 * heat_loss
        + 0.1 * stress_loss
        + 0.05 * porosity_loss
        + 0.05 * geometry_loss
    )

    losses = [total_loss, heat_loss, stress_loss, porosity_loss, geometry_loss]
    for loss in losses:
        assert torch.isfinite(loss), f"Loss {loss} is not finite"
        assert loss.numel() == 1
