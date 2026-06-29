# LPBF-Optimizer Roadmap

## Phase 1: Robustness and Validation (In Progress)

- [x] Fix physics loss to align with quality-metric outputs.
- [x] Fix NSGA-III objective direction.
- [x] Add tests and CI.
- [x] Refresh documentation and repository standards.
- [ ] Experimental validation campaign with XCT/EBSD/stress data.
- [ ] Calibrate synthetic data against real FEA/experiments.

## Phase 2: Melt Pool CFD and Grain Structure

- Integrate reduced-order CFD surrogate.
- Add grain-structure proxy models.

## Phase 3: Real-Time Data Assimilation and Control

- In-situ sensor integration stubs.
- Closed-loop parameter adaptation.

## Phase 4: Scan Path and Functionally Graded Materials

- Island/scan-strategy optimisation.
- Graded parameter fields.

## Phase 5: Full 3D Simulation and Microstructure Evolution

- High-fidelity 3D thermal-mechanical surrogate.
- Microstructure evolution coupling.
