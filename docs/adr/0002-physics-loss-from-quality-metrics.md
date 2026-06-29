# ADR 0002: Physics Loss Derived from Quality-Metric Outputs

## Status

Accepted

## Context

Because the PINN does not output temperature or a stress tensor, the original `physics.py` module double-concatenated inputs and misinterpreted outputs, causing runtime errors.

## Decision

Keep the three-output interface and compute physics-informed regularisation residuals:

- Heat residual from an analytic Rosenthal-like temperature field.
- Stress residual from Laplacian of predicted residual stress minus a thermo-elastic source.
- Porosity residual against an empirical energy-density indicator.
- Geometry residual against a nominal accuracy indicator based on temperature gradients.

## Consequences

- The network architecture stays unchanged.
- Residuals are simplified but physically motivated.
- Future work may add a separate temperature/stress output head if more rigorous PDE enforcement is needed.
