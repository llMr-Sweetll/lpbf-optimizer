"""Tests for uncertainty backends and predict_with_uncertainty."""

import pytest
import torch
from model import PINN
from uncertainty import DeepEnsembleBackend, MCDropoutBackend, get_uncertainty_backend


@pytest.fixture
def model():
    return PINN(input_dim=10, output_dim=3, width=32, depth=2, dropout_rate=0.5)


@pytest.fixture
def x():
    return torch.randn(8, 10)


def test_mc_dropout_backend_restores_eval_mode(model, x):
    """Calling predict_with_uncertainty must restore the original eval mode."""
    model.eval()
    assert not model.training

    mean, std = model.predict_with_uncertainty(x, num_samples=10)

    assert not model.training, "Model should remain in eval mode after MC Dropout"
    assert mean.shape == (8, 3)
    assert std.shape == (8, 3)


def test_mc_dropout_backend_restores_train_mode(model, x):
    """Calling predict_with_uncertainty must restore the original train mode."""
    model.train()
    assert model.training

    mean, std = model.predict_with_uncertainty(x, num_samples=10)

    assert model.training, "Model should be restored to train mode"
    assert mean.shape == (8, 3)
    assert std.shape == (8, 3)


def test_mc_dropout_only_enables_dropout_layers(model, x):
    """Only Dropout modules should be toggled, not the whole model."""
    model.eval()
    dropout_states_before = {
        id(m): m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)
    }

    model.predict_with_uncertainty(x, num_samples=5)

    dropout_states_after = {
        id(m): m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)
    }
    assert dropout_states_before == dropout_states_after


def test_get_uncertainty_backend():
    assert isinstance(get_uncertainty_backend("mc_dropout"), MCDropoutBackend)
    assert isinstance(get_uncertainty_backend("dropout"), MCDropoutBackend)
    assert isinstance(get_uncertainty_backend("ensemble"), DeepEnsembleBackend)
    assert isinstance(get_uncertainty_backend("deep_ensemble"), DeepEnsembleBackend)

    with pytest.raises(ValueError):
        get_uncertainty_backend("unknown")


def test_deep_ensemble_backend(model, x):
    ensemble = [model, model]
    backend = DeepEnsembleBackend()
    mean, std = backend.predict(ensemble, x)
    assert mean.shape == (8, 3)
    assert std.shape == (8, 3)


def test_deep_ensemble_backend_rejects_single_model(model, x):
    backend = DeepEnsembleBackend()
    with pytest.raises(TypeError):
        backend.predict(model, x)
