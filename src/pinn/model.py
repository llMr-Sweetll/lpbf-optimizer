import torch
import torch.nn as nn
from uncertainty import get_uncertainty_backend

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
        scaler_params (dict, optional): Standard-scaler parameters for input
            normalisation.  Expected structure::

                {
                    'scan_vectors': {'mean': [...], 'std': [...]},
                    'coordinates': {'mean': [...], 'std': [...]},
                }

            When provided, the scan-vector and coordinate slices of the input
            are normalised inside ``forward``; the time column is left unchanged.
            The mean/std tensors are saved in checkpoints via ``state_dict``.
        in_dim (int, optional): Backward-compatible alias for ``input_dim``.
        out_dim (int, optional): Backward-compatible alias for ``output_dim``.
    """
    def __init__(self, input_dim=10, output_dim=3, width=512, depth=5, dropout_rate=0.1,
                 apply_output_bounds=False, output_bounds=None,
                 output_bounds_temperature=100.0, scaler_params=None,
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
            scaler_params (dict, optional): Input normalisation parameters.
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

        # Register optional input normalisation parameters as non-trainable buffers.
        if scaler_params is not None:
            scan = scaler_params['scan_vectors']
            coord = scaler_params['coordinates']
            self.register_buffer(
                'scan_mean', torch.as_tensor(scan['mean'], dtype=torch.float32)
            )
            self.register_buffer(
                'scan_std', torch.as_tensor(scan['std'], dtype=torch.float32)
            )
            self.register_buffer(
                'coord_mean', torch.as_tensor(coord['mean'], dtype=torch.float32)
            )
            self.register_buffer(
                'coord_std', torch.as_tensor(coord['std'], dtype=torch.float32)
            )

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

    def _get_scan_dim(self):
        """Return the dimensionality of the scan-vector slice.

        The model assumes the trailing dimensions are ``[x, y, z, t]``
        (3 spatial coordinates + 1 time), so the scan-vector slice size is
        ``input_dim - 4``.
        """
        return self.input_dim - 4

    def _transform_inputs(self, x, inverse=False):
        """Apply or invert standard-scaler normalisation to input slices.

        The scan-vector and coordinate slices are scaled; the time column is
        left unchanged.  When no scaler parameters were provided, ``x`` is
        returned unchanged.

        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim].
            inverse (bool): If True, denormalise instead of normalise.

        Returns:
            torch.Tensor: Transformed input tensor with the same shape as ``x``.
        """
        if getattr(self, 'scan_mean', None) is None:
            return x

        scan_dim = self._get_scan_dim()
        sv = x[:, :scan_dim]
        coords = x[:, scan_dim:scan_dim + 3]
        t = x[:, scan_dim + 3:]

        scan_mean = self.scan_mean[:scan_dim]
        scan_std = self.scan_std[:scan_dim]

        if inverse:
            sv = sv * (scan_std + 1e-8) + scan_mean
            coords = coords * (self.coord_std + 1e-8) + self.coord_mean
        else:
            sv = (sv - scan_mean) / (scan_std + 1e-8)
            coords = (coords - self.coord_mean) / (self.coord_std + 1e-8)

        return torch.cat([sv, coords, t], dim=1)

    def transform_inputs(self, x):
        """Normalise the scan-vector and coordinate slices of ``x``.

        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim].

        Returns:
            torch.Tensor: Normalised input tensor.
        """
        return self._transform_inputs(x, inverse=False)

    def inverse_transform_inputs(self, x):
        """Invert input normalisation for the scan-vector and coordinate slices.

        Args:
            x (torch.Tensor): Normalised input tensor [batch_size, input_dim].

        Returns:
            torch.Tensor: Input tensor restored to original units.
        """
        return self._transform_inputs(x, inverse=True)

    def forward(self, S):
        """
        Forward pass through the network.

        Input normalisation is applied inside the model when scaler parameters
        were supplied at construction time, so callers can pass raw process
        parameters and coordinates.

        Args:
            S (torch.Tensor): Process parameter tensor [batch_size, input_dim]

        Returns:
            torch.Tensor: Predicted outcomes [batch_size, output_dim]
        """
        S = self._transform_inputs(S)
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

    def predict_with_uncertainty(self, x, num_samples=100, backend="mc_dropout"):
        """
        Estimate predictive uncertainty via a configurable backend.

        The default backend is Monte-Carlo (MC) Dropout (Gal & Ghahramani, 2016).
        Unlike the older implementation, only ``Dropout`` modules are enabled
        during inference; BatchNorm / LayerNorm layers are left untouched. This
        makes the method safe for future architectures that add normalisation
        layers.

        References:
            Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian
            Approximation: Representing Model Uncertainty in Deep Learning*. ICML.

        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim].
            num_samples (int): Number of stochastic forward passes. Ignored by
                backends that do not rely on repeated sampling (e.g. deep
                ensemble).
            backend (str): Backend name. Supported: ``mc_dropout``,
                ``deep_ensemble``.

        Returns:
            tuple: (mean_prediction [batch_size, output_dim],
                    std_deviation [batch_size, output_dim])
        """
        uncertainty_backend = get_uncertainty_backend(backend)
        return uncertainty_backend.predict(self, x, num_samples=num_samples)
