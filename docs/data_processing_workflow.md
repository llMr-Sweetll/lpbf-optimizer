# LPBF Data Processing Workflow

This document provides flowcharts and explanations of the data processing pipeline in the LPBF optimization framework.

## Overall Data Flow

```mermaid
flowchart TD
    A[Raw Data Sources] --> B[Data Preprocessing]
    B --> C[Training Dataset]
    C --> D[PINN Training]
    D --> E[Trained Surrogate Model]
    E --> F[Optimization]
    
    subgraph "Data Sources"
    A1[FEA Simulations] --> A
    A2[Experimental Measurements] --> A
    A3[Synthetic Data Generation] --> A
    end
    
    subgraph "Preprocessing Steps"
    B1[Data Cleaning] --> B
    B2[Feature Engineering] --> B
    B3[Normalization] --> B
    B4[Train/Val/Test Split] --> B
    end
```


## Data Structure

```mermaid
flowchart LR
    subgraph "Input Features"
    A1[Process Parameters]
    A2[Spatial Coordinates]
    A3[Time]
    end
    
    subgraph "Process Parameters"
    A1 --> P1[Laser Power P]
    A1 --> P2[Scan Speed v]
    A1 --> P3[Hatch Spacing h]
    A1 --> P4[Scan Angle θ]
    A1 --> P5[Island Size l_island]
    A1 --> P6[Layer Thickness]
    end
    
    subgraph "Output Labels"
    B1[Residual Stress σ_res]
    B2[Porosity φ_pore]
    B3[Geometric Accuracy Ratio GAR]
    end
    
    A1 & A2 & A3 --> C[PINN Model]
    C --> B1 & B2 & B3
```

## Data Preprocessing Pipeline

```mermaid
flowchart TD
    A[Raw Data] --> B[Data Loading]
    B --> C[Data Cleaning]
    C --> D[Feature Extraction]
    D --> E[Normalization]
    E --> F[Train/Val/Test Split]
    F --> G[HDF5 Dataset Creation]
    
    subgraph "Cleaning Operations"
    C1[Remove Outliers] --> C
    C2[Handle Missing Values] --> C
    C3[Filter Invalid Data] --> C
    end
    
    subgraph "Normalization Methods"
    E1[Min-Max Scaling] --> E
    E2[Z-Score Standardization] --> E
    end
```


## Synthetic Data Generation

```mermaid
flowchart LR
    A[Parameter Space Sampling] --> B[Physics-Based Equations]
    B --> C[Synthetic Dataset]
    
    D[Material Properties] --> B
    E[Noise Models] --> B
    
    subgraph "Sampling Methods"
    A1[Random Sampling] --> A
    A2[Latin Hypercube] --> A
    A3[Sobol Sequence] --> A
    end
    
    subgraph "Physics Models"
    B1[Heat Equation] --> B
    B2[Stress Equilibrium] --> B
    B3[Porosity Formation] --> B
    end
```


## Data Storage Format

```mermaid
flowchart TD
    A[HDF5 Dataset] --> B[Training Set]
    A --> C[Validation Set]
    A --> D[Test Set]
    
    subgraph "Dataset Structure"
    B1[Process Parameters] --> B
    B2[Coordinates] --> B
    B3[Time Points] --> B
    B4[Output Labels] --> B
    end
    
    subgraph "Metadata"
    A1[Normalization Parameters]
    A2[Material Properties]
    A3[Dataset Statistics]
    end
```


## Data Augmentation Techniques

```mermaid
flowchart LR
    A[Original Dataset] --> B[Data Augmentation]
    B --> C[Augmented Dataset]
    
    subgraph "Augmentation Methods"
    B1[Coordinate Transformation] --> B
    B2[Parameter Perturbation] --> B
    B3[Physics-Informed Interpolation] --> B
    end
```