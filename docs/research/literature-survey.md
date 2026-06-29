# Literature Survey: PINNs, UQ, and Optimisation for Laser Powder Bed Fusion

> A concise but comprehensive survey of the methods and open problems surrounding Physics-Informed Neural Networks (PINNs), uncertainty quantification (UQ), multi-objective Bayesian optimisation, digital twins, and CFD/grain-structure coupling for LPBF additive manufacturing. All references are collected in `docs/references.bib`.

## 1. Physics-Informed Neural Networks & Neural Operators for LPBF/AM Thermal and Melt-Pool Modelling

### 1.1 Foundational PINN ideas

The modern PINN framework embeds PDE residuals, boundary conditions and observed data into a single neural-network loss (Raissi et al., 2019). For additive manufacturing this is attractive because the governing heat equation is well known, while experimental labels are scarce or noisy. The core challenge is that LPBF involves moving Gaussian heat sources, phase change, steep thermal gradients and multiple spatial–temporal scales, all of which make standard PINN training difficult.

### 1.2 PINNs for thermal and melt-pool prediction

- **Zhu et al. (2021)** applied PINNs to predict temperature fields and melt-pool fluid dynamics in metal AM, showing that a physics-informed loss can compensate for limited labelled data.
- **Liao et al. (2023)** developed a hybrid thermal model that fuses sparse infrared-camera measurements with a PINN formulation to recover full-field temperatures and identify unknown material/process parameters.
- **Hosseini et al. (2023)** obtained a parametric single-track solution for LPBF via PINNs, demonstrating how one forward solve can be reused across laser powers and scan speeds.
- **Kim et al. (2025)** proposed a physics-guided deep generative model (MP-PG-CTGAN) that predicts future melt-pool monitoring images by encoding spatial–temporal physical relationships before the generative stage.
- **Sharma et al. (2025)** introduced FEA-PINN, a hybrid framework that periodically corrects PINN predictions with short FEA simulations to control error accumulation in long-duration thermal problems.
- **Depaoli et al. (2024)** used a parameterised PINN to replace repeated FEM runs during process-parameter optimisation, providing a first step toward data-driven LPBF optimisation.

### 1.3 Neural operators as fast surrogates

Neural operators learn mappings between function spaces rather than fixed grids, making them natural surrogates for parametric PDEs.

- **Lu et al. (2021)** introduced DeepONet, whose branch/trunk architecture can encode parametric source terms such as scan paths and laser schedules.
- **Li et al. (2021)** proposed the Fourier Neural Operator (FNO), which uses fast Fourier transforms to capture long-range correlations efficiently.
- **Kovachki et al. (2023)** unified these ideas in a general neural-operator framework and analysed their approximation properties.
- **Safari & Wessels (2025)** combined DeepONet with PINNs to predict 3-D temperature histories for multi-track LPBF scenarios, using a sequential PINN strategy to manage growing training complexity.
- **Liu et al. (2024)** built a deep-neural-operator digital twin for AM, fusing simulation and sensor data to predict thermal fields in near real time.
- **Kushwaha et al. (2024)** demonstrated advanced deep operator networks for multiphysics solution fields in materials processing, including AM-relevant thermal and mechanical problems.

### 1.4 Training pathologies and loss balancing

- **Wang et al. (2021)** analysed gradient flow pathologies in PINNs and showed that imbalanced back-propagated gradients cause biased training.
- **Bischof & Kraus (2021)** cast PINN loss balancing as a multi-objective optimisation problem, a perspective that directly motivates adaptive weighting strategies such as the GradNorm-style balancer used in this repository.

## 2. Residual Stress and Microstructure Prediction

Predicting residual stress, distortion and microstructure remains the central bridge from process parameters to part quality.

- **Sharma & Guo (2024)** developed a thermal–mechanical physics-informed deep-learning framework for fast prediction of thermal stress evolution in laser metal deposition.
- **Lestandi et al. (2024)** compared MLP, U-Net and interpolation-based surrogates for residual-stress prediction in LPBF, reporting sub-second inference with thousands-fold speed-ups over high-fidelity simulation.
- **Tang et al. (2024)** integrated PINNs with a cellular-automata (CA) microstructure model to calibrate thermo-microstructural simulations for LPBF.
- **Lach (2026)** reviewed recent CA and phase-field (PF) approaches for grain-scale microstructure evolution during metal AM, emphasising hybrid CA–PF frameworks that balance efficiency and thermodynamic fidelity.
- **Markl & Körner (2016)** gave a foundational multiscale perspective on powder-bed-based AM, linking powder-scale thermo-fluid dynamics to part-scale thermal histories.

## 3. Uncertainty Quantification

Reliable quality predictions require not only point estimates but also calibrated uncertainty, because LPBF data are noisy, physics is approximate, and optimisation must be robust to model error.

- **Gal & Ghahramani (2016)** showed that dropout at test time provides a scalable Bayesian approximation; this is the basis for the MC Dropout layer used in this repository's PINN.
- **Lakshminarayanan et al. (2017)** proposed deep ensembles as a simple, well-calibrated alternative that captures epistemic uncertainty by training multiple networks.
- **Kendall & Gal (2017)** separated aleatoric (data) and epistemic (model) uncertainty, a distinction that is essential when deciding whether to trust a prediction or collect more data.
- **Yang et al. (2021)** introduced B-PINNs, which place Bayesian neural networks inside the PINN loss and infer posterior distributions over PDE solutions.
- **Sensoy et al. (2018)** introduced evidential deep learning for classification uncertainty; recent extensions to regression offer an alternative to sampling-based methods.
- **Angelopoulos & Bates (2021)** provided a gentle introduction to conformal prediction, which can construct distribution-free prediction intervals around PINN or neural-operator outputs.
- **Psaros et al. (2023)** surveyed UQ methods in scientific machine learning, compared metrics and calibration, and highlighted the gap between theory and engineering practice.

## 4. Multi-Objective Bayesian Optimisation

LPBF process design is inherently many-objective (residual stress, porosity, geometric accuracy, productivity). Bayesian optimisation (BO) offers sample-efficient search when objectives are expensive to evaluate via simulation or experiment.

- **Deb & Jain (2014)** proposed NSGA-III, the reference-point-based many-objective evolutionary algorithm used in this repository for Pareto exploration.
- **Knowles (2006)** introduced ParEGO, an early BO method for expensive multi-objective problems that scalarises objectives via random Chebyshev weights.
- **Daulton et al. (2020)** derived a differentiable expected hypervolume improvement (qEHVI) acquisition function for parallel multi-objective BO.
- **Daulton et al. (2021)** extended this to the noisy setting with qNEHVI, which is now the default multi-objective acquisition in BoTorch.
- **Ament et al. (2023)** introduced the LogEI family (qLogEI, qLogEHVI, qLogNEHVI), reformulating classic acquisitions in log-space to avoid numerical pathologies during optimisation.
- **Balandat et al. (2020)** released BoTorch, the Monte-Carlo BO library that underpins Ax and provides the acquisition functions above.

## 5. In-Situ Monitoring, Digital Twins and Closed-Loop Control

Closing the loop between sensing, simulation and actuation is the next frontier for LPBF.

- **McCann et al. (2021)** reviewed in-situ sensing, process monitoring and machine control in LPBF, identifying pyrometry, photodiode, camera and acoustic sensors as the dominant data streams.
- **Bevans et al. (2024)** proposed a digital-twin architecture for rapid in-situ qualification of LPBF part quality, fusing heterogeneous sensor data with physics-based signatures.
- **Riensche et al. (2025a)** developed a physics-guided, layer-wise thermal-history controller for LPBF and showed across four geometries that controlling thermal history improves grain uniformity, geometric accuracy and surface finish.
- **Riensche et al. (2025b)** introduced DynamicPrint, a feedforward model-predictive control framework that adjusts laser power and speed layer-by-layer before printing to mitigate thermal-induced defects.
- **Wang et al. (2023)** built a customised LPBF platform with real-time melt-pool monitoring and closed-loop laser-power control, demonstrating stable melt-pool signatures across overhang and thin-wall features.
- **Fang et al. (2024)** surveyed process monitoring, diagnosis and control of AM from a systems perspective, organising the literature into defect detection, machine fault diagnosis and closed-loop control loops.

## 6. CFD / Grain-Structure Coupling

Melt-pool fluid flow, vaporisation, spatter and solidification microstructure are tightly coupled, yet they span length scales from micrometres to part scale.

- **Vinuesa & Brunton (2022)** reviewed how machine learning can enhance CFD, covering turbulence closure, reduced-order modelling and data-driven correction of coarse simulations.
- **Lach (2026)** surveyed CA and PF methods for microstructure evolution, noting that hybrid CA–PF schemes allocate PF to solidification fronts and CA to bulk grain competition.
- **Tang et al. (2024)** showed how PINN-derived temperature histories can drive a CA microstructure model, providing a blueprint for physics-informed CFD-to-microstructure coupling.
- **Markl & Körner (2016)** emphasised that bridging powder-scale CFD, mesoscale melt-pool dynamics and part-scale thermal stress is still the grand challenge in AM simulation.

## 7. Open Research Directions and Relevance to this Repository

The LPBF-Optimizer prototype sits at the intersection of several active research frontiers. The literature points to the following priorities for extending the current codebase:

1. **Surrogate fidelity.** Move from the current fully-connected PINN toward neural-operator surrogates (DeepONet/FNO) or FEA-PINN corrections to handle multi-track, multi-layer thermal histories more accurately (Safari & Wessels, 2025; Sharma et al., 2025; Liu et al., 2024).
2. **Uncertainty-aware optimisation.** Replace or augment MC Dropout with deep ensembles and conformal prediction so that Pareto fronts and Bayesian acquisition functions reflect genuine model confidence (Lakshminarayanan et al., 2017; Angelopoulos & Bates, 2021; Psaros et al., 2023).
3. **Advanced multi-objective BO.** Adopt qLogEHVI/qLogNEHVI via BoTorch/Ax for sample-efficient Pareto search, especially when the surrogate is expensive to evaluate (Ament et al., 2023; Daulton et al., 2021).
4. **Closed-loop experimental validation.** Integrate in-situ sensor streams and digital-twin concepts from Bevans et al. (2024), Riensche et al. (2025a,b) and Wang et al. (2023) to move from open-loop optimisation toward adaptive control.
5. **CFD and microstructure coupling.** Add melt-pool CFD and CA/PF grain-structure modules so that the surrogate predicts not only thermal fields but also porosity and grain morphology (Tang et al., 2024; Lach, 2026; Markl & Körner, 2016).
6. **Loss balancing and training robustness.** Continue to apply adaptive multi-objective loss balancing as the physics loss grows, following Bischof & Kraus (2021) and Wang et al. (2021).

By aligning the repository's PINN surrogate, NSGA-III/Bayesian optimisers and validation stubs with these directions, the project can evolve from a research prototype into a credible digital-twin backbone for LPBF process design.
