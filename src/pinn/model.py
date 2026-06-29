import torch
import torch.nn as nn


# Default physical ranges for the LPBF quality metrics predicted by the model.
# These bounds are used when ``apply_output_bounds=True`` and no custom bounds
# are supplied.  The ranges match the synthetic data generator and are wide
# enough to cover typical Ti-6Al-4V LPBF parts.
_DEFAULT_OUTPUT_BOUNDS = (
    (50.0, 800.0),    # residual stress (MPa)
    (0.0, 0.30),      # porosity (volume fraction)
    (0.7, 1.0),       # geometric accuracy (ratio)
)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for predicting LPBF process outcomes
    (residual stress, porosity, geometric accuracy) from process parameters.

    Args:
        input_dim (int): Total input dimension (process parameters + spatial coordinates + time)
        output_dim (int): Output dimension (number of predicted outcomes)
        width (int): Width of hidden layers
        depth (int): Number of hidden layers
        dropout_rate (float): Dropout rate for MC Dropout.
        apply_output_bounds (bool): If True, squash each output into a physical
            range via a sigmoid transform.
        output_bounds (sequence of tuple, optional): ``(min, max)`` pairs for
            each output dimension.  Defaults to ``_DEFAULT_OUTPUT_BOUNDS`` when
            ``apply_output_bounds`` is True.
        output_bounds_temperature (float): Temperature for the sigmoid bound
            transform.  A higher value keeps the transform near-linear in the
            central region, which stabilises training (default 100.0).
        in_dim (int, optional): Backward-compatible alias for ``input_dim``.
        out_dim (int, optional): Backward-compatible alias for ``output_dim``.
    """
    def __init__(self, input_dim=10, output_dim=3, width=512, depth=5, dropout_rate=0.1,
                 apply_output_bounds=False, output_bounds=None,
                 output_bounds_temperature=100.0,
                 in_dim=None, out_dim=None):
        """
        Initialize the PINN model.

        Args:
            input_dim (int): Total input dimension (process parameters + spatial coordinates + time)
            output_dim (int): Output dimension (number of predicted outcomes)
            width (int): Width of hidden layers
            depth (int): Number of hidden layers
            dropout_rate (float): Dropout rate for MC Dropout (Gal & Ghahramani, 2016).
                                Default 0.1 is common for regression tasks.
            apply_output_bounds (bool): If True, constrain outputs to physical ranges.
            output_bounds (sequence of tuple, optional): Per-output (min, max) bounds.
            output_bounds_temperature (float): Sigmoid temperature for bounded outputs.
            in_dim (int, optional): Backward-compatible alias for ``input_dim``.
            out_dim (int, optional): Backward-compatible alias for ``output_dim``.
        """
        super().__init__()

        # Support legacy ``in_dim`` / ``out_dim`` argument names as aliases for
        # the names used in ``data/params.yaml``. If an alias is explicitly
        # provided, it takes precedence for backward compatibility.
        if in_dim is not None:
            input_dim = in_dim
        if out_dim is not None:
            output_dim = out_dim

        self.input_dim = input_dim  # Store input dimension
        # Preserve legacy attribute for callers that rely on it.
        self.in_dim = input_dim

        self.apply_output_bounds = apply_output_bounds
        self.output_bounds_temperature = output_bounds_temperature
        if apply_output_bounds:
            bounds = output_bounds if output_bounds is not None else _DEFAULT_OUTPUT_BOUNDS
            if len(bounds) != output_dim:
                raise ValueError(
                    f"output_bounds length ({len(bounds)}) must match output_dim ({output_dim})"
                )
            mins = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
            maxs = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
            self.register_buffer('output_mins', mins)
            self.register_buffer('output_maxs', maxs)
        else:
            self.register_buffer('output_mins', None)
            self.register_buffer('output_maxs', None)

        # Build the network with SiLU activation and Dropout
        layers = []
        layers.append(nn.Linear(input_dim, width))

        for _ in range(depth-1):
            layers += [
                nn.SiLU(),
                nn.Dropout(p=dropout_rate),  # Monte Carlo Dropout layer
                nn.Linear(width, width)
            ]

        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(width, output_dim)

    def forward(self, S):
        """
        Forward pass through the network.

        Args:
            S (torch.Tensor): Process parameter tensor [batch_size, input_dim]

        Returns:
            torch.Tensor: Predicted outcomes [batch_size, output_dim]
        """
        raw = self.out(self.hidden(S))
        if self.apply_output_bounds:
            # Map each raw logit to the configured [min, max] interval.  The
            # temperature keeps the transform near-linear around the midpoint,
            # avoiding gradient pathology when the network starts close to zero.
            return self.output_mins + (self.output_maxs - self.output_mins) * torch.sigmoid(raw / self.output_bounds_temperature)
        return raw

    def get_output_bounds(self):
        """
        Return the per-output physical bounds.

        Returns:
            tuple: ``((min_0, max_0), ...)`` when bounds are enabled, else ``None``.
        """
        if not self.apply_output_bounds:
            return None
        return tuple(
            (float(self.output_mins[i]), float(self.output_maxs[i]))
            for i in range(len(self.output_mins))
        )

    def predict_with_uncertainty(self, x, num_samples=100):
        """
        Perform Monte Carlo Dropout inference to estimate predictive uncertainty.

        Reference:
            Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. ICML.

            For relevance to Additive Manufacturing (AM):
            [1] Zhao, Mirihanage, et al. (2025). "Revealing melt flow instabilities in LPBF".
            [2] "Physics-Informed Neural Networks for Additive Manufacturing: A Review" (2025).

        Args:
            x (torch.Tensor): Input tensor
            num_samples (int): Number of stochastic forward passes

        Returns:
            tuple: (mean_prediction, std_deviation)
        """
        # Enable dropout during inference while preserving the original mode so
        # the caller's model state is unchanged after MC Dropout sampling.
        was_training = self.training
        self.train()

        try:
            predictions = []
            with torch.no_grad():
                for _ in range(num_samples):
                    predictions.append(self.forward(x).unsqueeze(0))

            # Stack predictions: [num_samples, batch_size, out_dim]
            predictions = torch.cat(predictions, dim=0)

            # Calculate mean and standard deviation (aleatoric + epistemic uncertainty approx)
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)
        finally:
            self.train(was_training)

        return mean_pred, std_pred
