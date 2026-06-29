import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "pinn"))

import torch
import torch.nn.functional as F
from loss_balancer import AdaptiveLossBalancer


def test_weights_sum_to_num_losses():
    balancer = AdaptiveLossBalancer(num_losses=3, alpha=1.5)
    net = torch.nn.Linear(4, 3)
    x = torch.rand(3, 4)
    pred = net(x)
    target = torch.zeros(3, 3)
    losses = [F.mse_loss(pred[:, i], target[:, i]) for i in range(3)]

    weights = balancer.update_weights(losses, net.weight)
    assert torch.isclose(weights.sum(), torch.tensor(3.0), atol=1e-4)
