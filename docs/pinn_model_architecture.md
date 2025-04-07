# LPBF PINN Model Architecture

This document provides flowcharts and explanations of the Physics-Informed Neural Network (PINN) architecture used in the LPBF optimization framework.

## PINN Model Overview

```mermaid
flowchart TD
    A[Input Layer] --> B[Hidden Layers]
    B --> C[Output Layer]
    D[Physics Constraints] --> E[Loss Function]
    C --> E
    F[Training Data] --> E
    E --> G[Backpropagation]
    G --> B
    
    subgraph "Input Features"
    A1[Process Parameters] --> A
    A2[Spatial Coordinates] --> A
    A3[Time] --> A
    end
    
    subgraph "Physics Constraints"
    D1[Heat Equation] --> D
    D2[Stress Equilibrium] --> D
    end
    
    subgraph "Output Predictions"
    C1[Residual Stress] --> C
    C2[Porosity] --> C
    C3[Geometric Accuracy] --> C
    end
```

## Neural Network Architecture

```mermaid
flowchart LR
    A[Input Layer] --> B1[Hidden Layer 1]
    B1 --> B2[Hidden Layer 2]
    B2 --> B3[Hidden Layer 3]
    B3 --> B4[Hidden Layer 4]
    B4 --> B5[Hidden Layer 5]
    B5 --> C[Output Layer]
    
    subgraph "Layer Details"
    A1["Input (9 neurons)\nProcess params + coords + time"] --> A
    B["Hidden (512 neurons each)\nSiLU Activation"] --> B1 & B2 & B3 & B4 & B5
    C1["Output (3 neurons)\nResidual stress, porosity, geometric accuracy"] --> C
    end
```

## PINN Training Process

```mermaid
flowchart TD
    A[Initialize PINN] --> B[Forward Pass]
    B --> C[Compute Data Loss]
    B --> D[Compute Physics Loss]
    C & D --> E[Total Loss]
    E --> F[Backpropagation]
    F --> G[Update Weights]
    G --> H{Convergence?}
    H -->|No| B
    H -->|Yes| I[Trained Model]
    
    subgraph "Data Loss Components"
    C1[MSE: Predicted vs. Actual Residual Stress] --> C
    C2[MSE: Predicted vs. Actual Porosity] --> C
    C3[MSE: Predicted vs. Actual Geometric Accuracy] --> C
    end
    
    subgraph "Physics Loss Components"
    D1[Heat Equation Residual] --> D
    D2[Stress Equilibrium Residual] --> D
    end
```

## Automatic Differentiation for Physics Constraints

```mermaid
flowchart LR
    A[PINN Predictions] --> B[Compute Gradients]
    B --> C[Physics Residuals]
    
    subgraph "Spatial Derivatives"
    B1["∇T (Temperature Gradient)"] --> B
    B2["∇²T (Temperature Laplacian)"] --> B
    B3["∇·σ (Stress Divergence)"] --> B
    end
    
    subgraph "Temporal Derivatives"
    B4["∂T/∂t (Temperature Rate)"] --> B
    B5["∂fs/∂t (Solidification Rate)"] --> B
    end
    
    C --> D[Physics Loss Terms]
    D --> E[Total Loss Function]
```

## Loss Function Components

```mermaid
flowchart TD
    A[Total Loss] --> B[Data Loss]
    A --> C[Physics Loss]
    
    B --> B1[MSE Loss]
    
    C --> C1[Heat Equation Residual]
    C --> C2[Stress Equilibrium Residual]
    
    subgraph "Loss Weighting"
    D1["λ_data (Data Weight)"] --> B
    D2["λ_heat (Heat Equation Weight)"] --> C1
    D3["λ_stress (Stress Equation Weight)"] --> C2
    end
    
    B & C1 & C2 --> E["L = L_data + λ_heat·L_heat + λ_stress·L_stress"]
```

## Training Hyperparameters

```mermaid
flowchart LR
    subgraph "Optimizer Settings"
    A1[Adam Optimizer]
    A2[Learning Rate: 0.001]
    A3[Weight Decay: 1.0e-5]
    end
    
    subgraph "Scheduler Settings"
    B1[ReduceLROnPlateau]
    B2[Factor: 0.5]
    B3[Patience: 20]
    end
    
    subgraph "Training Parameters"
    C1[Epochs: 500]
    C2[Batch Size: 64]
    C3[Gradient Clipping: 1.0]
    end
    
    subgraph "Physics Weights"
    D1[λ_heat: 0.1]
    D2[λ_stress: 0.1]
    end
```