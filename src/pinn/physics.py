import torch


def _analytic_temperature(S, coords, t, mat_props):
    """Rosenthal-like analytic temperature rise from a moving point source.

    Coordinates and beam radius are in mm, while conductivity ``k`` is supplied
    in W/m·K. We convert to W/mm·K for a dimensionally consistent temperature
    rise: ``T = T0 + eta*P / (2*pi*k_eff*r) * exp(-2*r^2/r0^2)``.

    Args:
        S (torch.Tensor): Process parameters [N, n_params].
        coords (torch.Tensor): Spatial coordinates [N, 3], requires_grad=True.
        t (torch.Tensor): Time [N, 1], requires_grad=True.
        mat_props (dict): Material properties.

    Returns:
        torch.Tensor: Temperature field [N, 1].
    """
    P = S[:, 0:1]
    v = S[:, 1:2]
    eta = mat_props['eta']
    k = mat_props['k']          # W/m·K
    r0 = mat_props['r0']        # mm

    k_eff = k / 1000.0          # W/mm·K

    x_laser = v * t
    r_squared = (coords[:, 0:1] - x_laser) ** 2 + coords[:, 1:2] ** 2 + coords[:, 2:3] ** 2
    r = torch.sqrt(r_squared + 1e-8)

    # Gaussian-damped moving point-source approximation
    T0 = 300.0
    T = T0 + (eta * P) / (2.0 * torch.pi * k_eff * r) * torch.exp(-2.0 * r_squared / r0 ** 2)
    return T


def _laplacian(field, coords):
    """Compute the Laplacian of `field` with respect to `coords`."""
    grad_field = torch.autograd.grad(
        field.sum(), coords, create_graph=True, retain_graph=True
    )[0]
    laplacian = 0.0
    for i in range(coords.shape[1]):
        second = torch.autograd.grad(
            grad_field[:, i].sum(), coords, create_graph=True, retain_graph=True
        )[0][:, i]
        laplacian = laplacian + second
    return laplacian


def _heat_residual(T, S, coords, t, mat_props):
    """Transient heat equation residual for the analytic temperature field.

    All quantities are expressed in a consistent mm-based unit system:
      - ``rho`` (kg/m^3) -> ``rho_mm`` (kg/mm^3)
      - ``k`` (W/m·K) -> ``k_mm`` (W/mm·K)
      - ``cp`` and ``Hm`` remain in J/kg·K and J/kg, respectively.

    The laser heat flux ``q_laser`` (W/mm^2) is converted to an effective
    volumetric source using a characteristic melt-pool penetration depth of
    0.1 mm, so that the residual is in W/mm^3.
    """
    rho = mat_props['rho']
    cp = mat_props['cp']
    k = mat_props['k']
    Hm = mat_props['Hm']
    Ts = mat_props['Ts']
    Tl = mat_props['Tl']
    eta = mat_props['eta']
    r0 = mat_props['r0']

    # Convert to mm-based units.
    rho_mm = rho / 1e9
    k_mm = k / 1000.0
    penetration_depth = 0.1  # mm

    dTdt = torch.autograd.grad(T.sum(), t, create_graph=True, retain_graph=True)[0]
    laplacian_T = _laplacian(T, coords)

    # Latent heat phase-change term
    fs = 0.5 * (1.0 + torch.tanh((Tl - T) / (Tl - Ts) * 10.0))
    dfsdT = torch.autograd.grad(fs.sum(), T, create_graph=True, retain_graph=True)[0]
    dfsdt = dfsdT * dTdt

    # Gaussian laser source (surface flux, W/mm^2)
    P = S[:, 0:1]
    v = S[:, 1:2]
    x_laser = v * t
    r_squared = (coords[:, 0:1] - x_laser) ** 2 + coords[:, 1:2] ** 2 + coords[:, 2:3] ** 2
    q_laser = 2.0 * eta * P / (torch.pi * r0 ** 2) * torch.exp(-2.0 * r_squared / r0 ** 2)

    # Effective volumetric source (W/mm^3).
    q_vol = q_laser / penetration_depth

    residual = rho_mm * cp * dTdt - k_mm * laplacian_T - q_vol + rho_mm * Hm * dfsdt
    return residual


def _stress_residual(sigma, T, coords, mat_props):
    """Thermo-elastic equilibrium residual for the predicted residual stress.

    The predicted residual stress ``sigma`` is a scalar field in MPa. Its
    Laplacian therefore has units of MPa/mm^2. The thermo-elastic source term
    is ``E_MPa * alpha / (1 - nu) * laplacian(T)``, which is also in MPa/mm^2.
    """
    E_GPa = mat_props['E']         # Young's modulus (GPa)
    alpha = mat_props['alpha']     # Thermal expansion coefficient (1/K)
    nu = mat_props['nu']           # Poisson's ratio

    E_MPa = E_GPa * 1000.0         # Convert GPa -> MPa

    laplacian_sigma_MPa = _laplacian(sigma, coords)
    laplacian_T = _laplacian(T, coords)

    source_MPa = (E_MPa * alpha / (1.0 - nu)) * laplacian_T
    residual = laplacian_sigma_MPa - source_MPa
    return residual


def _porosity_residual(porosity, S):
    """Porosity residual against an empirical energy-density indicator."""
    P = S[:, 0]
    v = S[:, 1]
    h = S[:, 2]
    # Use the last parameter column as layer thickness rather than hardcoding
    # an index, matching the parameter order defined in the config.
    layer_thickness = S[:, -1]
    ed = P / (v * h * layer_thickness + 1e-8)
    # Normalize to a reference energy density of ~10 J/mm^3 for the current
    # parameter bounds and keep the indicator bounded via a sigmoid.
    ed_norm = ed / 10.0
    porosity_indicator = 0.05 + 0.2 * torch.sigmoid((ed_norm - 1.5) * 2.0)
    residual = porosity.squeeze() - porosity_indicator
    return residual


def _geometry_residual(accuracy, T, coords):
    """Geometric accuracy residual against a nominal accuracy indicator."""
    grad_T = torch.autograd.grad(
        T.sum(), coords, create_graph=True, retain_graph=True
    )[0]
    grad_T_mag = torch.sqrt(torch.sum(grad_T ** 2, dim=1) + 1e-8)
    accuracy_indicator = 1.0 - 0.05 * torch.tanh(grad_T_mag / 1000.0)
    accuracy_indicator = torch.clamp(accuracy_indicator, 0.7, 1.0)
    residual = accuracy.squeeze() - accuracy_indicator
    return residual


def compute_physics_loss(
    model,
    S_batch,
    coords_batch,
    t_batch,
    mat_props,
    lambda_heat=1.0,
    lambda_stress=1.0,
    lambda_porosity=0.1,
    lambda_geometry=0.1,
    return_components=False,
    use_predicted_temperature=False,
):
    """Compute physics-informed regularisation losses aligned with model outputs.

    The PINN predicts [residual_stress, porosity, geometric_accuracy].  This
    function computes:
      - heat residual from an analytic temperature field,
      - stress-equilibrium residual from the predicted residual stress,
      - porosity residual from an empirical energy-density indicator,
      - geometric accuracy residual from a nominal accuracy indicator.

    Args:
        model: PINN model.
        S_batch (torch.Tensor): Process parameters [N, n_params].
        coords_batch (torch.Tensor): Spatial coordinates [N, 3].
        t_batch (torch.Tensor): Time [N, 1].
        mat_props (dict): Material properties.
        lambda_heat (float): Weight for heat residual.
        lambda_stress (float): Weight for stress residual.
        lambda_porosity (float): Weight for porosity residual.
        lambda_geometry (float): Weight for geometry residual.
        return_components (bool): If True, returns individual losses.
        use_predicted_temperature (bool): If True, use the model's temperature
            head for the heat residual instead of the analytic field.

    Returns:
        torch.Tensor or tuple: Total physics loss, or (heat, stress, porosity, geometry) losses.
    """
    # Enable autograd on spatial and temporal inputs for residual computation.
    coords = coords_batch.clone().requires_grad_(True)
    t = t_batch.clone().requires_grad_(True)

    # Forward pass with differentiable inputs.
    model_input = torch.cat([S_batch, coords, t], dim=1)
    if use_predicted_temperature:
        if not getattr(model, "predict_temperature", False):
            raise ValueError(
                "use_predicted_temperature=True requires a model with "
                "predict_temperature=True"
            )
        out, T_pred = model(model_input, return_temperature=True)
        T = T_pred
    else:
        out = model(model_input)
        T = _analytic_temperature(S_batch, coords, t, mat_props)

    residual_stress = out[:, 0:1]
    porosity = out[:, 1:2]
    geometric_accuracy = out[:, 2:3]

    heat_res = _heat_residual(T, S_batch, coords, t, mat_props)
    heat_loss = torch.mean(heat_res ** 2)

    stress_res = _stress_residual(residual_stress, T, coords, mat_props)
    stress_loss = torch.mean(stress_res ** 2)

    porosity_res = _porosity_residual(porosity, S_batch)
    porosity_loss = torch.mean(porosity_res ** 2)

    geometry_res = _geometry_residual(geometric_accuracy, T, coords)
    geometry_loss = torch.mean(geometry_res ** 2)

    if return_components:
        return heat_loss, stress_loss, porosity_loss, geometry_loss

    total = (
        lambda_heat * heat_loss
        + lambda_stress * stress_loss
        + lambda_porosity * porosity_loss
        + lambda_geometry * geometry_loss
    )
    return total
