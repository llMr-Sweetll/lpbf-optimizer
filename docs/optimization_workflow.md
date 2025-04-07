# LPBF Optimization Workflow

This document provides flowcharts and explanations of the optimization process in the LPBF framework, showing how optimal process parameters are determined.

## Overall Optimization Workflow

```mermaid
flowchart TD
    A[Trained PINN Model] --> B[Define Optimization Problem]
    B --> C[Multi-Objective Optimization]
    C --> D[Pareto-Optimal Solutions]
    D --> E[Solution Selection]
    E --> F[Experimental Validation]
    F -->|Feedback Loop| A
    
subgraph Problem_Definition [Problem Definition]
B1[Parameter Bounds] --> B
B2[Objective Functions] --> B
B3[Constraints] --> B
end

subgraph Optimization_Algorithms [Optimization Algorithms]
C1[NSGA-III] --> C
C2[Bayesian Optimization] --> C
end
```

## Multi-Objective Optimization with NSGA-III

```mermaid
flowchart TD
    A[Initialize Population] --> B[Evaluate Objectives]
    B --> C[Non-dominated Sorting]
    C --> D[Reference Point Association]
    D --> E[Selection]
    E --> F[Crossover & Mutation]
    F --> G{Convergence?}
    G -->|No| B
    G -->|Yes| H[Pareto Front]
    
subgraph Objective_Evaluation [Objective Evaluation]
B1[PINN Surrogate Model] --> B
end

subgraph NSGA3_Parameters [NSGA-III Parameters]
I1[Population Size: 100]
I2[Generations: 100]
I3[Reference Points: 12]
end
```

## Surrogate Problem Definition

```mermaid
flowchart LR
    A[Process Parameters] --> B[PINN Surrogate Model]
    B --> C[Objective Values]
    
    subgraph "Process Parameters"
    A1[Laser Power P] --> A
    A2[Scan Speed v] --> A
    A3[Hatch Spacing h] --> A
    A4[Scan Angle θ] --> A
    A5[Island Size l_island] --> A
    A6[Layer Thickness] --> A
    end
    
    subgraph "Objective Functions"
    C1[Residual Stress] --> C
    C2[Porosity] --> C
    C3[Geometric Accuracy] --> C
    end
    
    subgraph "Parameter Bounds"
    D1["P: 150-400 W"]
    D2["v: 100-2000 mm/s"]
    D3["h: 0.05-0.15 mm"]
    D4["θ: 0-90°"]
    D5["l_island: 2-10 mm"]
    D6["layer_thickness: 0.02-0.06 mm"]
    end
```

## Pareto Front Analysis

```mermaid
flowchart TD
    A[Pareto-Optimal Solutions] --> B[Trade-off Analysis]
    B --> C[Solution Clustering]
    C --> D[Representative Solutions]
    D --> E[Final Selection]
    
    subgraph "Selection Criteria"
    E1[Manufacturing Constraints] --> E
    E2[Design Requirements] --> E
    E3[Process Stability] --> E
    end
```

## Bayesian Optimization Alternative

```mermaid
flowchart LR
    A[Initial Design Points] --> B[Evaluate Objectives]
    B --> C[Build Surrogate Model]
    C --> D[Acquisition Function]
    D --> E[Propose Next Point]
    E --> F[Evaluate New Point]
    F --> G{Convergence?}
    G -->|No| C
    G -->|Yes| H[Optimal Solution]
    
    subgraph "Surrogate Models"
    C1[Gaussian Process] --> C
    end
    
    subgraph "Acquisition Functions"
    D1[Expected Improvement] --> D
    D2[Upper Confidence Bound] --> D
    D3[Probability of Improvement] --> D
    end
```

## Experimental Validation Loop

```mermaid
flowchart TD
    A[Optimized Parameters] --> B[Build Test Coupons]
    B --> C[Characterization]
    C --> D[Compare Predictions vs. Measurements]
    D --> E{Acceptable Error?}
    E -->|No| F[Update Training Data]
    F --> G[Retrain PINN Model]
    G --> H[Re-optimize]
    E -->|Yes| I[Final Process Parameters]
    
    subgraph "Characterization Methods"
    C1[X-ray CT for Porosity] --> C
    C2[XRD for Residual Stress] --> C
    C3[CMM for Geometric Accuracy] --> C
    end
```