# LPBF Scientific Workflow

This document provides flowcharts and explanations of the scientific processes involved in Laser Powder Bed Fusion (LPBF) optimization.

## Physics-Based Modeling Workflow

```mermaid
flowchart TD
    A[Laser-Material Interaction] --> B[Heat Transfer]
    B --> C[Phase Change]
    C --> D[Thermal Expansion]
    D --> E[Residual Stress Development]
    E --> F[Mechanical Deformation]
    F --> G[Porosity Formation]
    G --> H[Final Part Properties]
    
    subgraph "Energy Input"
    A
    end
    
    subgraph "Thermal Processes"
    B
    C
    end
    
    subgraph "Mechanical Processes"
    D
    E
    F
    end
    
    subgraph "Quality Metrics"
    G
    H
    end
```

## Heat Transfer Equation

The heat transfer in LPBF is modeled using the heat equation with a moving laser source:

```mermaid
flowchart LR
    A[Heat Equation] --> B["ρc_p∂T/∂t = ∇·(k∇T) + Q - H_m∂f_s/∂t"]
    
    C["Laser Heat Source\nQ = 2ηP/(πr_0²)exp(-2r²/r_0²)"] --> B
    D["Material Properties\nρ, c_p, k, H_m"] --> B
    E["Phase Change\nf_s = f(T, T_s, T_l)"] --> B
```

Where:
- ρ: Density (kg/m³)
- c_p: Specific heat capacity (J/kg·K)
- T: Temperature (K)
- t: Time (s)
- k: Thermal conductivity (W/m·K)
- η: Laser absorption coefficient
- P: Laser power (W)
- r_0: Laser beam radius (mm)
- r: Distance from laser center (mm)
- H_m: Latent heat of melting (J/kg)
- f_s: Solid fraction

## Residual Stress Development

```mermaid
flowchart TD
    A[Temperature Field] --> B[Thermal Strain]
    B --> C[Mechanical Strain]
    C --> D[Stress Field]
    D --> E[Residual Stress]
    
    F["Constitutive Equation\nσ = C:ε^e"] --> D
    G["Equilibrium Equation\n∇·σ = 0"] --> D
    H["Plastic Flow\nε̇^p = A(σ_eq/σ_y)^n"] --> C
```

Where:
- σ: Stress tensor
- C: Elasticity tensor
- ε^e: Elastic strain tensor
- ε^p: Plastic strain tensor
- σ_eq: Equivalent stress
- σ_y: Yield strength
- A, n: Material constants

## Process-Structure-Property Relationships

```mermaid
flowchart LR
    subgraph "Process Parameters"
    A[Laser Power]
    B[Scan Speed]
    C[Hatch Spacing]
    D[Scan Strategy]
    E[Layer Thickness]
    end
    
    subgraph "Microstructure"
    F[Grain Size]
    G[Phase Composition]
    H[Porosity]
    I[Melt Pool Geometry]
    end
    
    subgraph "Properties"
    J[Residual Stress]
    K[Mechanical Strength]
    L[Geometric Accuracy]
    M[Surface Roughness]
    end
    
    A & B & C & D & E --> F & G & H & I
    F & G & H & I --> J & K & L & M
```

## Energy Density Relationship

```mermaid
flowchart TD
    A[Laser Power P] --> E["Volumetric Energy Density\nE = P/(v·h·t)"]  
    B[Scan Speed v] --> E
    C[Hatch Spacing h] --> E
    D[Layer Thickness t] --> E
    
    E --> F[Melt Pool Characteristics]
    F --> G[Part Quality]
```