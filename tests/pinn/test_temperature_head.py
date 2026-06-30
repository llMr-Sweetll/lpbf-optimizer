"""Tests for the optional trainable temperature output head."""

import pytest
import torch
from model import PINN
from physics import compute_physics_loss
from train import PINNTrainer


@pytest.fixture
def x():
    return torch.randn(8, 10)


def test_model_without_temperature_head(x):
    model = PINN(input_dim=10, output_dim=3, width=32, depth=2, predict_temperature=False)
    out = model(x)
    assert out.shape == (8, 3)
    assert not hasattr(model, 'temperature_head') or model.temperature_head is None


def test_model_with_temperature_head(x):
    model = PINN(input_dim=10, output_dim=3, width=32, depth=2, predict_temperature=True)
    out = model(x)
    assert out.shape == (8, 3)

    out, T = model(x, return_temperature=True)
    assert out.shape == (8, 3)
    assert T.shape == (8, 1)


def test_predicted_temperature_in_physics_loss():
    model = PINN(input_dim=10, output_dim=3, width=32, depth=2, predict_temperature=True)
    model.train()

    S = torch.randn(8, 6, requires_grad=False)
    coords = torch.randn(8, 3, requires_grad=True)
    t = torch.randn(8, 1, requires_grad=True)
    mat_props = {
        'rho': 4430.0, 'cp': 526.3, 'k': 6.7, 'eta': 0.35, 'r0': 0.05,
        'Hm': 286000.0, 'Ts': 1604.0, 'Tl': 1660.0,
        'E': 110.0, 'nu': 0.34, 'alpha': 8.6e-6,
    }

    loss = compute_physics_loss(
        model, S, coords, t, mat_props,
        use_predicted_temperature=True,
    )
    assert loss.item() >= 0.0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.temperature_head.parameters())


def test_trainer_temperature_loss(tmp_path):
    """A single training step with predict_temperature should produce gradients."""
    config = {
        'model': {
            'input_dim': 10,
            'output_dim': 3,
            'hidden_width': 16,
            'hidden_depth': 2,
            'dropout_rate': 0.0,
            'apply_output_bounds': False,
            'predict_temperature': True,
        },
        'data': {
            'processed_data_path': str(tmp_path / 'data.h5'),
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
        },
        'training': {
            'n_epochs': 1,
            'batch_size': 8,
            'random_seed': 42,
            'device': 'cpu',
            'optimizer': {'type': 'adam', 'learning_rate': 0.01, 'weight_decay': 0.0},
            'scheduler': {'type': 'none'},
            'lambda_heat': 0.1,
            'lambda_stress': 0.0,
            'lambda_porosity': 0.0,
            'lambda_geometry': 0.0,
            'clip_grad': False,
            'checkpoint_freq': 100,
            'output_dir': str(tmp_path / 'models'),
            'plot_freq': 100,
            'print_freq': 100,
        },
        'material_properties': {
            'rho': 4430.0, 'cp': 526.3, 'k': 6.7, 'eta': 0.35, 'r0': 0.05,
            'Hm': 286000.0, 'Ts': 1604.0, 'Tl': 1660.0,
            'E': 110.0, 'nu': 0.34, 'alpha': 8.6e-6,
        },
    }

    # Create a tiny synthetic HDF5 dataset
    from pathlib import Path

    import h5py
    import numpy as np
    Path(config['data']['processed_data_path']).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(config['data']['processed_data_path'], 'w') as f:
        f.create_dataset('train/scan_vectors', data=np.random.rand(16, 6).astype('float32'))
        f.create_dataset('train/coordinates', data=np.random.rand(16, 3).astype('float32'))
        f.create_dataset('train/time', data=np.random.rand(16, 1).astype('float32'))
        f.create_dataset('train/outputs', data=np.random.rand(16, 3).astype('float32'))
        f.create_dataset('val/scan_vectors', data=np.random.rand(4, 6).astype('float32'))
        f.create_dataset('val/coordinates', data=np.random.rand(4, 3).astype('float32'))
        f.create_dataset('val/time', data=np.random.rand(4, 1).astype('float32'))
        f.create_dataset('val/outputs', data=np.random.rand(4, 3).astype('float32'))
        f.create_dataset('test/scan_vectors', data=np.random.rand(4, 6).astype('float32'))
        f.create_dataset('test/coordinates', data=np.random.rand(4, 3).astype('float32'))
        f.create_dataset('test/time', data=np.random.rand(4, 1).astype('float32'))
        f.create_dataset('test/outputs', data=np.random.rand(4, 3).astype('float32'))

    trainer = PINNTrainer(config_path=None, config=config, num_threads=1)
    trainer.load_data()

    S, coords, t, y = next(iter(trainer.train_loader))
    losses = trainer.train_step(S, coords, t, y)
    assert losses['temperature'] > 0.0
    assert losses['total'] > 0.0

    # Check that the temperature head has gradients
    assert any(p.grad is not None for p in trainer.model.temperature_head.parameters())
