# LPBF Optimizer Development Roadmap

## Phase 1: Robustness & Validation (Current)

- [x] **Core Architecture**: Data Generation -> PINN Training -> NSGA-III Optimization (Verified)
- [ ] **Experimental Validation Workflow**:
  - [ ] Design DOE for physical coupon printing (varying P, v, hatch).
  - [ ] Correlate predicted porosity/stress with XRD/CT-scan data.
  - [ ] Calibrate PINN physics parameters ($\eta$, $\lambda_{heat}$) against experimental ground truth.
- [ ] **Uncertainty Quantification (UQ)**:
  - [ ] Implement Bayesian PINN (BNN) layers to output aleatoric uncertainty ($\sigma$).
  - [ ] Quantify epistemic uncertainty via Deep Ensembles for high-stakes aerospace confidence intervals.

## Phase 2: Multi-Scale Physics Integration

- [ ] **Melt Pool Dynamics (Mesoscale)**:
  - [ ] Integrate Computational Fluid Dynamics (CFD) loss terms (Navier-Stokes) to model Marangoni convection.
  - [ ] Simulate keyhole mode vs. conduction mode transition boundaries.
- [ ] **Grain Structure Evolution (Microscale)**:
  - [ ] Couple PINN thermal history with Cellular Automata (CA) or Phase Field models.
  - [ ] Predict grain size distribution, orientation (texture), and columnar-to-equiaxed transition (CET).
  - [ ] **Surrogate Modeling**: Train Graph Neural Networks (GNNs) on Phase Field simulation data to predict grain structure in real-time.

## Phase 3: Digital Twin & In-Situ Control

- [ ] **Real-Time Data Assimilation**:
  - [ ] Integrate sensor streams (pyrometer, high-speed camera, photodiode).
  - [ ] Develop Kalman Filter or 4D-Var data assimilation to update PINN state variables on-the-fly.
- [ ] **Closed-Loop "Self-Healing" Control**:
  - [ ] Implement Model Predictive Control (MPC) using the PINN digital twin.
  - [ ] Dynamically adjust laser power/speed to maintain constant melt pool depth despite complex geometry thermal buildup.
  - [ ] Detect and correct lack-of-fusion defects layer-by-layer.

## Phase 4: Advanced Material & Geometry Intelligence

- [ ] **Scan Path Optimization**:
  - [ ] Move beyond simple raster vectors (Skywriting, Fractal patterns).
  - [ ] Optimize local dwell times for geometric feature fidelity (overhangs, thin walls).
- [ ] **Functionally Graded Materials (FGMs)**:
  - [ ] Optimize parameters for transitioning alloy compositions (e.g., SS316L to Inconel 718).
  - [ ] Model mixing kinetics and intermetallic phase formation risks.
