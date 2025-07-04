# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project provides a comprehensive framework for analyzing neural networks through topological data analysis (TDA). It computes persistent homology on neural network activations, decision boundaries, and network architectures themselves during training. The framework supports multiple training backends, homology computation methods, and visualization tools.

## Key Commands

### Running the Main Pipeline
```bash
# Full pipeline execution (creates conda env, installs deps, runs main.py)
./run_main.sh

# Direct execution (requires manual environment setup)
python main.py
```

### Testing
```bash
# Run all tests with proper environment setup
./run_tests.sh

# Run specific test
pytest tests/test_pipeline.py -v
```

### Environment Setup
```bash
# Create conda environment with dependencies
conda create -y -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**IMPORTANT: Always activate the conda environment before running any scripts:**

Due to conda shell integration issues, use the direct path method:
```bash
# Use myenv Python directly
/opt/anaconda3/envs/myenv/bin/python script_name.py

# Or if conda activation works in your shell:
conda activate myenv
python script_name.py
```

All Python scripts in this repository must be run within the `myenv` conda environment.
The myenv Python path is: `/opt/anaconda3/envs/myenv/bin/python`

## Architecture Overview

### Core Pipeline Flow
1. **Data Generation** (`src/data/`): Creates synthetic torus datasets, MNIST data, or custom datasets
2. **Neural Network Training** (`src/models/`): Trains networks with multiple backend options
3. **Topological Analysis** (`src/topology/`): Computes persistent homology using three methods
4. **Visualization** (`src/visualization/`): Creates Betti curves, decision boundaries, and network graphs

### Training Implementations

#### PyTorch Backends
- **`torch_mlp.py`**: Standard PyTorch MLP with batch normalization, dropout, mixed precision
- **`torch_vectorized.py`**: Vectorized implementation for efficient batch processing
- **`torch_parallel.py`**: Parallel training of multiple networks using multiprocessing
- **`torch_custom.py`**: Custom architectures with variable depth and width
- **`trainer_with_homology.py`**: Tracks network homology evolution during training
- **`decision_boundary_trainer.py`**: Extracts and analyzes decision boundaries

#### Apple Silicon (MLX) Backends
- **`trainer_mlx.py`**: Core MLX implementation for Apple Silicon
- **`trainer_mlx_parallel.py`**: Parallel MLX training
- **`vectorised_mlx.py`**: Vectorized MLX operations
- **`mlx_simple_mlp.py`**: Simplified MLX model

### Homology Computation Methods

1. **GUDHI** (`compute_homology.py`): Standard persistent homology computation
2. **Ripser** (`compute_homology_ripser.py`): Fast parallel implementation
3. **Witness Complex** (`compute_homology_witness.py`): Memory-efficient for large datasets

### Specialized Pipelines

#### Network Topology Analysis
- **Network Homology Tracking**: Analyzes the evolving graph structure of neural networks
- **Decision Boundary Topology**: Computes homology of decision boundaries in 3D
- **Wasserstein Generalization**: Measures topological differences between train/test sets

#### Visualization Tools
- **Betti Curves**: Statistical analysis with confidence intervals
- **3D Decision Boundaries**: Interactive boundary visualization with animations
- **Network Graphs**: Directed graph visualization of network architecture
- **UMAP Projections**: Dimensionality reduction for activation spaces

### Configuration System
- **`training_config.yaml`**: Model architecture, training hyperparameters
- **`homology_config.yaml`**: Persistent homology parameters (method, dimensions, thresholds)
- **`visualization_config.yaml`**: Plotting settings and output formats
- **`decision_boundary_config.yaml`**: Boundary extraction parameters
- **`network_homology_config.yaml`**: Network graph analysis settings
- **`optimization_config.yaml`**: Hyperparameter optimization with Optuna
- **`search_config.yaml`**: Grid search parameters

### Key Entry Points
- **`main.py`**: Primary pipeline orchestrator
- **`run_optimization.sh`**: Launch hyperparameter optimization
- **`launch_dashboard.sh`**: Start Optuna dashboard for monitoring
- **Specialized scripts**: Various analysis and visualization entry points

### Device Support
- **CPU**: Standard computation
- **CUDA**: NVIDIA GPU acceleration
- **MPS**: Apple Silicon GPU via Metal Performance Shaders
- **MLX**: Apple's optimized ML framework

Device selection via `device` parameter in configs ('auto', 'cpu', 'cuda', 'mps').

### Output Structure
```
results/
├── models/           # Trained model checkpoints
├── plots/            # Visualizations and Betti curves
├── homology/         # Persistence diagrams and barcodes
├── boundaries/       # Decision boundary data
├── network_graphs/   # Network topology evolution
└── optimization/     # Hyperparameter search results
```

### Advanced Features
- **Hyperparameter Optimization**: Automated tuning with Optuna dashboard
- **Parallel Processing**: Multi-network training and analysis
- **Memory Optimization**: Witness complex for large-scale datasets
- **Real-time Tracking**: Monitor topology changes during training
- **Distance Metrics**: Multiple persistence diagram comparison methods

## Common Workflows and Pipelines

### 1. Standard Activation Homology Pipeline
```bash
# Run the complete pipeline: data → training → homology → visualization
./run_main.sh
```
This executes the standard workflow analyzing layer activations.

### 2. Decision Boundary Analysis
```python
# Extract and analyze decision boundaries
python src/models/decision_boundary_trainer.py
python src/visualization/visualize_decision_boundaries.py
```
Creates 3D visualizations of how decision boundaries evolve during training.

### 3. Network Graph Homology
```python
# Track the homology of the neural network graph structure
python src/topology/network_homology_tracker.py
```
Analyzes how the network's graph topology changes during training.

### 4. Hyperparameter Optimization
```bash
# Launch optimization with Optuna
./run_optimization.sh

# Monitor in real-time with dashboard
./launch_dashboard.sh
```
Automated search for optimal homology computation parameters.

### 5. Parallel Multi-Network Analysis
```python
# Train multiple networks in parallel
python src/models/torch_parallel.py

# Track homology across multiple networks
python src/topology/track_homology.py
```
Statistical analysis across multiple training runs.

### 6. Wasserstein Generalization Analysis
```python
# Analyze topological differences between train/test
python src/analysis/wasserstein_generalization.py
```
Measures how well topological features generalize.

### 7. Custom Dataset Visualization
```python
# 2D visualization
python src/data/visualize_dataset.py

# Interactive 3D visualization
python src/data/visualize_dataset_3d.py
```

### 8. Parameter Grid Search
```python
# Find optimal homology parameters
python src/utils/parameter_grid_search.py  # GUDHI
python src/utils/parameter_grid_search_ripser.py  # Ripser
```

### 9. Network Homology Comparison
```python
# Compare homology distances between multiple trained networks
python src/analysis/network_homology_comparison.py

# With custom settings
python src/analysis/network_homology_comparison.py --max-models 10 --models-dir results/custom_models
```
Computes pairwise distances between network graph homologies and generates heatmaps and clustering analysis.

## Tips for Development

1. **Memory Management**: Use witness complex for datasets > 10k points
2. **Speed Optimization**: Use Ripser for faster computation, especially with parallel processing
3. **Device Selection**: Let 'auto' choose the best available device
4. **Debugging**: Check intermediate outputs in `results/` subdirectories
5. **Configuration**: Start with provided YAML files and adjust parameters incrementally

## External Research Project

The `nn-evolution/` directory contains a separate project for monitoring neural network learning without validation sets. It requires significant computational resources (1.5TB RAM for full experiments).