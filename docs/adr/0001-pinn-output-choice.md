# ADR 0001: PINN Predicts Quality Metrics, Not Temperature/Stress Fields

## Status

Accepted

## Context

LPBF process modelling can predict either raw physical fields (temperature, melt-pool velocity, stress tensor) or downstream quality metrics (residual stress magnitude, porosity, geometric accuracy).

## Decision

The PINN predicts three quality metrics:

- residual stress (MPa)
- porosity (%)
- geometric accuracy (ratio)

## Consequences

- The model is directly usable for optimisation without post-processing.
- Physics loss must be formulated as a regularisation on these outputs rather than a full field PDE residual.
- Input dimension remains fixed at 10 (6 parameters + 3 coords + 1 time).
