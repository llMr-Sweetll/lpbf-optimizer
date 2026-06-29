# ADR 0004: Synthetic Data as Default Training Source

## Status

Accepted

## Context

Real FEA data requires ABAQUS/COMSOL licenses and long runtimes. Experimental data is expensive.

## Decision

Provide a physics-inspired synthetic data generator as the default training source. FEA and experimental data can be ingested via `src/preprocessing.py` and `src/validate/characterise.py` when available.

## Consequences

- New users can run the full pipeline immediately.
- Synthetic data is not a substitute for validated physics; results are illustrative.
