# ADR 0003: NSGA-III as Default Optimiser

## Status

Accepted

## Context

The project requires multi-objective optimisation over conflicting quality metrics.

## Decision

Use NSGA-III as the default optimiser because it handles three objectives well and is straightforward to wrap around a deterministic surrogate. Bayesian optimisation is provided as a single-objective alternative via Ax/BoTorch.

## Consequences

- NSGA-III returns a Pareto front in one run.
- Bayesian optimisation requires separate runs per objective or more complex MOO extensions.
