# LPBF Optimizer Flowchart Documentation

This document serves as an index to the flowchart documentation for the LPBF (Laser Powder Bed Fusion) Optimizer project. The flowcharts provide visual representations of the key processes, data flows, and algorithms used in the system.

## Available Documentation

### 1. [Scientific Workflow](scientific_workflow.md)

This document explains the physics-based modeling approach used in LPBF optimization:
- Physics-based modeling workflow from laser-material interaction to final part properties
- Heat transfer equation with moving laser source
- Residual stress development process
- Process-structure-property relationships
- Energy density relationships

### 2. [Data Processing Workflow](data_processing_workflow.md)

This document details the data pipeline from raw inputs to processed outputs:
- Overall data flow from various sources to the optimization stage
- Data structure for inputs and outputs
- Preprocessing pipeline steps
- Synthetic data generation process
- Data storage format and organization
- Data augmentation techniques

### 3. [PINN Model Architecture](pinn_model_architecture.md)

This document explains the neural network structure and physics integration:
- PINN model overview showing inputs, outputs, and physics constraints
- Detailed neural network architecture
- Training process with data and physics loss components
- Automatic differentiation for physics constraints
- Loss function components and weighting
- Training hyperparameters

### 4. [Optimization Workflow](optimization_workflow.md)

This document describes how optimal process parameters are determined:
- Overall optimization workflow from trained model to experimental validation
- Multi-objective optimization with NSGA-III algorithm
- Surrogate problem definition with parameters and objectives
- Pareto front analysis for solution selection
- Bayesian optimization alternative approach
- Experimental validation loop

## How to Use This Documentation

These flowcharts are designed to help you understand the complex processes involved in LPBF optimization. They can be used for:

1. **Onboarding new team members** - Providing a visual overview of the system
2. **Understanding system components** - Breaking down complex processes into manageable parts
3. **Identifying process relationships** - Seeing how different components interact
4. **Troubleshooting issues** - Tracing data and process flows to locate problems

## Viewing the Flowcharts

The flowcharts are created using Mermaid markdown syntax. To view them properly:

1. Use a Markdown viewer that supports Mermaid diagrams (like GitHub, VS Code with extensions, or specialized Markdown editors)
2. Alternatively, copy the Mermaid code blocks into the [Mermaid Live Editor](https://mermaid.live/) for interactive viewing

## Contributing to Documentation

If you need to update or extend these flowcharts:

1. Follow the existing Mermaid syntax patterns
2. Ensure diagrams remain clear and focused on a single aspect of the system
3. Update the overview document when adding new flowchart files