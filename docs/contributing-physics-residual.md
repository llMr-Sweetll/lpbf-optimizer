# Contributing a New Physics Residual

This guide shows how to extend the PINN loss in `src/pinn/physics.py` with a new physics-informed residual. A residual is a soft constraint that penalises predictions which violate a physical law or empirical relationship.

## Overview

The physics loss is computed in `compute_physics_loss(model, S_batch, coords_batch, t_batch, mat_props, ...)`. It currently combines:

- **Heat residual** — transient heat equation on the temperature field.
- **Stress residual** — thermo-elastic equilibrium on the predicted residual stress.
- **Porosity residual** — porosity vs. energy-density indicator.
- **Geometry residual** — geometric accuracy vs. thermal gradient indicator.

Adding a new residual follows three steps:

1. Implement the residual function.
2. Register it in `compute_physics_loss`.
3. Add a unit test and wire it into the config.

---

## Step 1: Implement the residual function

Create a private helper in `src/pinn/physics.py` that accepts the relevant fields and returns a per-sample residual tensor. For example, a simple surface-roughness residual that penalises roughness when the thermal gradient is high:

```python
def _roughness_residual(roughness, T, coords):
    """Example residual: roughness grows with thermal gradient magnitude.

    Args:
        roughness (torch.Tensor): Predicted surface roughness [N, 1].
        T (torch.Tensor): Temperature field [N, 1].
        coords (torch.Tensor): Spatial coordinates [N, 3], requires_grad=True.

    Returns:
        torch.Tensor: Residual [N,].
    """
    grad_T = torch.autograd.grad(
        T.sum(), coords, create_graph=True, retain_graph=True
    )[0]
    grad_T_mag = torch.sqrt(torch.sum(grad_T ** 2, dim=1) + 1e-8)
    roughness_indicator = 1.0 + 0.01 * grad_T_mag
    return roughness.squeeze() - roughness_indicator
```

### Conventions

- Use `torch.autograd.grad(..., create_graph=True, retain_graph=True)` so gradients flow back into the network.
- Accept coordinate tensors that already have `requires_grad=True`.
- Return a tensor that can be squared and averaged: `loss = torch.mean(residual ** 2)`.

---

## Step 2: Register the residual

Add a `lambda_<name>` argument to `compute_physics_loss` and include the new term in the total loss.

```python
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
    lambda_roughness=0.0,  # new
    return_components=False,
    use_predicted_temperature=False,
):
```

Inside the function, after the existing residuals, add:

```python
# Example: roughness is an extra model output at index 3
roughness = out[:, 3:4]
roughness_res = _roughness_residual(roughness, T, coords)
roughness_loss = torch.mean(roughness_res ** 2)
```

Update the total loss and the component return:

```python
total = (
    lambda_heat * heat_loss
    + lambda_stress * stress_loss
    + lambda_porosity * porosity_loss
    + lambda_geometry * geometry_loss
    + lambda_roughness * roughness_loss
)

if return_components:
    return heat_loss, stress_loss, porosity_loss, geometry_loss, roughness_loss

return total
```

> **Note:** if you change the number of returned components, update the callers in `src/pinn/train.py` and any tests accordingly.

---

## Step 3: Add the weight to the config

Open `data/params.yaml` and add the new weight under `training`:

```yaml
training:
  lambda_roughness: 0.0
```

In `src/pinn/train.py`, read the weight in `train_step` and pass it to `compute_physics_loss`.

---

## Step 4: Test the residual

Create a test in `tests/pinn/test_physics.py` (or a new file) that:

1. Builds a tiny model with the required output dimension.
2. Calls `compute_physics_loss` with `return_components=True`.
3. Asserts the new component is non-negative and has a finite gradient.

```python
def test_roughness_residual():
    model = PINN(input_dim=10, output_dim=4, width=32, depth=2)
    S = torch.randn(8, 6)
    coords = torch.randn(8, 3, requires_grad=True)
    t = torch.randn(8, 1, requires_grad=True)
    mat_props = { ... }
    losses = compute_physics_loss(
        model, S, coords, t, mat_props,
        lambda_roughness=1.0,
        return_components=True,
    )
    assert len(losses) == 5
    assert losses[-1].item() >= 0.0
    losses[-1].backward()
    assert any(p.grad is not None for p in model.parameters())
```

### Testing checklist

- [ ] Residual tensor has the correct shape.
- [ ] Loss is non-negative and finite.
- [ ] Backpropagation reaches model parameters.
- [ ] Setting the lambda weight to zero removes the term from the total loss.
- [ ] New config key is documented in `data/params.yaml`.
- [ ] `ruff check .` is clean.
- [ ] `pytest tests/pinn/ -q` passes.

---

## Need help?

Open a discussion or issue on GitHub with:

- The physical law you want to encode.
- The inputs and outputs involved.
- Any literature references.
