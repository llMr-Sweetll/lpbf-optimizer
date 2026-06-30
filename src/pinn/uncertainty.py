"""Pluggable uncertainty backends for the LPBF PINN surrogate.

The default backend uses Monte-Carlo (MC) Dropout (Gal & Ghahramani, 2016). To
keep the implementation safe for future architectures that may include
BatchNorm or LayerNorm, the MC Dropout backend only enables ``Dropout`` modules
during inference instead of putting the whole model into training mode.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# Dropout module types that should be active during MC sampling.
_DROPOUT_MODULES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


class BaseUncertaintyBackend(ABC):
    """Base class for uncertainty estimation backends."""

    @abstractmethod
    def predict(self, model, x, num_samples=100):
        """Return (mean, std) for ``model(x)``.

        Args:
            model (torch.nn.Module): Surrogate model.
            x (torch.Tensor): Input tensor [batch_size, input_dim].
            num_samples (int): Number of forward passes to aggregate.

        Returns:
            tuple: (mean_prediction [batch_size, output_dim],
                    std_prediction [batch_size, output_dim])
        """
        raise NotImplementedError


class MCDropoutBackend(BaseUncertaintyBackend):
    """Monte-Carlo Dropout uncertainty backend.

    Only ``Dropout`` layers are switched on during inference; all other layers
    (e.g. BatchNorm) remain in their original mode.
    """

    def predict(self, model, x, num_samples=100):
        """Run ``num_samples`` stochastic forward passes and return mean/std."""
        # Capture original training state of Dropout modules *before* changing
        # the global model mode.
        was_training = model.training
        dropout_states = [
            (module, module.training)
            for module in model.modules()
            if isinstance(module, _DROPOUT_MODULES)
        ]

        # Keep BatchNorm/LayerNorm in eval mode while enabling only Dropout.
        model.eval()
        for module, _ in dropout_states:
            module.train()

        try:
            predictions = []
            with torch.no_grad():
                for _ in range(num_samples):
                    predictions.append(model(x).unsqueeze(0))
            predictions = torch.cat(predictions, dim=0)
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)
        finally:
            # Restore the global model mode first, then fix individual Dropout
            # modules to their original states.
            model.train(was_training)
            for module, state in dropout_states:
                module.train(state)

        return mean_pred, std_pred


class DeepEnsembleBackend(BaseUncertaintyBackend):
    """Deep-ensemble uncertainty backend.

    Accepts a list of models and aggregates their predictions. This is a
    future-facing interface; the repository currently trains a single model.
    """

    def predict(self, model, x, num_samples=100):
        """Run ensemble inference.

        Args:
            model (list[torch.nn.Module]): Ensemble member models.
            x (torch.Tensor): Input tensor.
            num_samples (int): Ignored for deep ensembles.

        Returns:
            tuple: (mean_prediction, std_prediction)
        """
        if not isinstance(model, (list, tuple)):
            raise TypeError(
                "DeepEnsembleBackend expects a list of models, got "
                f"{type(model).__name__}"
            )
        if not model:
            raise ValueError("Deep ensemble must contain at least one model")

        with torch.no_grad():
            predictions = torch.stack([m(x) for m in model], dim=0)
        return predictions.mean(dim=0), predictions.std(dim=0)


_BACKEND_REGISTRY = {
    "mc_dropout": MCDropoutBackend,
    "dropout": MCDropoutBackend,
    "ensemble": DeepEnsembleBackend,
    "deep_ensemble": DeepEnsembleBackend,
}


def get_uncertainty_backend(name):
    """Return an uncertainty backend instance by name.

    Args:
        name (str): One of ``mc_dropout``, ``ensemble``.

    Returns:
        BaseUncertaintyBackend: Instantiated backend.
    """
    name = name.lower().replace("-", "_")
    if name not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown uncertainty backend '{name}'. "
            f"Available: {list(_BACKEND_REGISTRY.keys())}"
        )
    return _BACKEND_REGISTRY[name]()
