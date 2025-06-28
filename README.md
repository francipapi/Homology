# Neural Network Topological Analysis

This project analyzes the topological properties of neural networks during training by computing persistent homology on layer activations and decision boundaries. The framework provides a comprehensive pipeline for understanding how neural networks learn through the lens of algebraic topology.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Description](#pipeline-description)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
- [Results and Visualization](#results-and-visualization)
- [Testing](#testing)

## Overview

The project combines machine learning with topological data analysis (TDA) to:
- Track topological changes in neural network activations during training
- Analyze decision boundary evolution and complexity
- Compute persistent homology to quantify topological features
- Visualize Betti curves showing topological invariants across layers
- Support multiple training backends (PyTorch, MLX for Apple Silicon)
- Enable parallel and vectorized training for efficiency

## Project Structure

```
Homology/
├── main.py                 # Main pipeline orchestrator
├── configs/               # Configuration files
│   ├── training_config.yaml
│   ├── homology_config.yaml
│   ├── visualization_config.yaml
│   └── decision_boundary_config.yaml
├── src/
│   ├── data/             # Data generation and processing
│   ├── models/           # Neural network implementations
│   ├── topology/         # Persistent homology computation
│   ├── visualization/    # Plotting and visualization tools
│   ├── utils/           # Utility functions
│   └── analysis/        # Analysis tools
├── tests/               # Test suite
├── results/             # Output directory (auto-created)
└── scripts/             # Utility shell scripts
```

## Installation

### Prerequisites
- Python 3.9+
- Conda (recommended) or pip
- CUDA toolkit (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Homology
```

2. Create and activate conda environment:
```bash
conda create -y -n myenv python=3.9
conda activate myenv
```

3. Install dependencies:
```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Quick Start

### Run the Complete Pipeline
```bash
# Using the provided script (handles environment setup)
./run_main.sh

# Or directly with Python (requires activated environment)
python main.py
```

### Run Tests
```bash
./run_tests.sh
```

## Pipeline Description

### 1. Data Generation (`src/data/`)

The pipeline begins by generating synthetic datasets, primarily torus pairs (linked/unlinked):

- **`dataset.py`**: Core module for torus generation
  - `generate()`: Creates complex torus pair datasets with configurable parameters
  - `gen_easy()`: Simplified torus generation for testing
  - `farthest_point_sampling()`: Reduces point cloud size while preserving structure
  
- **Visualization tools**:
  - `visualize_dataset.py`: 2D projections of datasets
  - `visualize_dataset_3d.py`: Interactive 3D visualizations

### 2. Neural Network Training (`src/models/`)

Multiple training paradigms are supported:

#### Standard PyTorch Training
- **`torch_mlp.py`**: Modern MLP implementation with batch normalization and dropout
- **`trainer.py`** (in `old_models/`): Simple trainer used by main pipeline
- Features: Early stopping, activation extraction, device flexibility (CPU/CUDA/MPS)

#### Parallel Training
- **`torch_parallel.py`**: Trains multiple networks concurrently using multiprocessing
- Ideal for ensemble studies or statistical analysis across multiple runs

#### Vectorized Training
- **`torch_vectorized.py`**: Efficient simultaneous training of multiple networks
- Uses PyTorch's `vmap` for hardware-optimized parallel computation

#### Decision Boundary Extraction
- **`decision_boundary_trainer.py`**: Specialized trainer that extracts decision boundaries during training
- Captures boundary evolution at specified epoch intervals

#### Apple Silicon Support (MLX)
- Various MLX implementations in `old_models/` for M1/M2/M3 optimization
- Leverages unified memory architecture for efficient computation

### 3. Topological Analysis (`src/topology/`)

Computes persistent homology on extracted activations:

- **`homology.py`**: Main implementation using GUDHI
  - `compute_persistent_homology()`: Core computation function
  - Outputs: persistence diagrams, barcodes, Betti numbers
  
- **Alternative implementations**:
  - `homology_ripser.py`: Ripser-based computation (often faster)
  - `witness_torch.py`: PyTorch-based witness complex computation
  - `compute_boundary_homology.py`: Specialized for decision boundaries

### 4. Visualization (`src/visualization/`)

Rich visualization capabilities for analysis results:

- **`plot_curves.py`**: Betti curve visualization across layers
- **`betti_curves.py`**: Advanced statistical analysis of Betti numbers
- **`decision_boundary_viz.py`**: 3D decision boundary evolution
- **`uma_plot.py`**: UMAP-based dimensionality reduction plots

### 5. Utilities (`src/utils/`)

Supporting functions for the pipeline:

- **`graph.py`**: Distance matrix computation for Rips complex
- **`distance_computation.py`**: Various distance metrics
- **`parameter_grid_search.py`**: Hyperparameter optimization
- **`compute_torus_homology_*.py`**: Ground truth homology computation

## Configuration

The pipeline is controlled via YAML configuration files in `configs/`:

### `training_config.yaml`
```yaml
model:
  width: 100          # Hidden layer width
  layers: 5           # Number of layers
  
training:
  epochs: 200
  batch_size: 32
  learning_rate: 0.001
  device: 'auto'      # 'cpu', 'cuda', 'mps', or 'auto'
  
data:
  generation:
    n: 1000           # Points per torus
    big_radius: 10
    small_radius: 3
```

### `homology_config.yaml`
```yaml
computation:
  max_dimension: 2    # Maximum homology dimension
  max_edge_length: 3.0
  num_neighbors: 50   # For k-NN graph construction
```

### `decision_boundary_config.yaml`
```yaml
extraction:
  grid:
    resolution: [100, 100, 100]
  boundary_detection:
    threshold: 0.5
    tolerance: 0.01
```

## Advanced Features

### Decision Boundary Analysis

Track how decision boundaries evolve during training:

```python
# Enable in decision_boundary_config.yaml
training:
  extraction_schedule:
    enabled: true
    frequency: 5  # Extract every 5 epochs
```

### Parallel Training

Train multiple networks simultaneously:

```python
# In training_config.yaml
num_networks: 10
max_parallel_workers: 4
```

### Hyperparameter Optimization

```bash
./run_optimization.sh
```

## Results and Visualization

Results are organized in the `results/` directory:

```
results/
├── models/          # Trained model checkpoints
├── plots/           # Dataset and Betti curve visualizations
├── homology/        # Persistence diagrams and Betti numbers
└── decision_boundaries/  # Boundary evolution data
```

### Key Outputs

1. **Betti Curves**: Show topological complexity across network layers
2. **Persistence Diagrams**: Visualize birth-death of topological features
3. **Decision Boundaries**: 3D visualizations of classification surfaces
4. **Training Metrics**: Loss curves, accuracy, and convergence analysis

## Testing

Run the test suite to verify installation:

```bash
# All tests
./run_tests.sh

# Specific test
pytest tests/test_pipeline.py -v

# Boundary extraction test
python test_boundary_training.py
```

## Advanced Usage

### Custom Datasets

Implement custom data generation:

```python
from src.data.dataset import DatasetGenerator

def custom_dataset(n_points):
    # Your implementation
    return X, y
```

### Adding New Topological Features

Extend the homology computation:

```python
from src.topology.homology import BaseHomology

class CustomHomology(BaseHomology):
    def compute(self, data):
        # Your implementation
        pass
```

### Batch Processing

For large-scale experiments:

```bash
# Configure multiple runs in configs/
python -m src.utils.batch_runner
```

## Troubleshooting

### Common Issues

1. **CUDA/MPS not available**: Set `device: 'cpu'` in `training_config.yaml`
2. **Memory errors**: Reduce `batch_size` or `grid.resolution`
3. **Import errors**: Ensure `PYTHONPATH` includes project root

### Performance Tips

- Use `torch_vectorized.py` for multiple network training
- Enable `cache_predictions` in `decision_boundary_config.yaml`
- Adjust `num_workers` for optimal parallelization

## Citation

If you use this code in your research, please cite:

```bibtex
@software{neural_topology_analysis,
  title={Neural Network Topological Analysis},
  author={Your Name},
  year={2024},
  url={repository-url}
}
```

## License

[Specify your license here]

## Acknowledgments

This project uses:
- [GUDHI](https://gudhi.inria.fr/) for topological computations
- [PyTorch](https://pytorch.org/) for neural network training
- [Plotly](https://plotly.com/) for interactive visualizations