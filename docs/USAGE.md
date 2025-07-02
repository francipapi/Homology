# Complete Usage Guide - Neural Network Topological Analysis

This comprehensive guide covers all pipelines, scripts, and configurations in the Homology repository. It consolidates information from all documentation files (except README.md and CLAUDE.md) into a single reference.

## Table of Contents
1. [Main Pipelines](#main-pipelines)
2. [Data Generation](#data-generation)
3. [Model Training](#model-training)
4. [Topological Analysis](#topological-analysis)
5. [Network Graph Homology](#network-graph-homology)
6. [Visualization](#visualization)
7. [Configuration Files](#configuration-files)
8. [Utilities and Tools](#utilities-and-tools)
9. [Testing](#testing)
10. [Advanced Features](#advanced-features)

## Main Pipelines

### 1. Complete Analysis Pipeline (`main.py`)

The main orchestrator that runs the entire analysis workflow.

```bash
# Basic usage (uses default configs)
python main.py

# With custom configs
python main.py --training-config configs/my_training.yaml \
               --homology-config configs/my_homology.yaml
```

**Pipeline steps:**
1. Generate torus dataset
2. Train neural network
3. Extract layer activations
4. Compute persistent homology
5. Visualize results

### 2. Network Homology Pipeline

Analyzes neural network architecture as a graph during training.

```bash
# Train with network homology tracking
python src/models/trainer_with_homology.py \
    --training-config configs/training_config.yaml \
    --homology-config configs/network_homology_config.yaml \
    --model-type custom  # or "mlp"
```

### 3. Decision Boundary Pipeline

Extracts and analyzes decision boundaries during training.

```bash
# Run decision boundary training
python src/models/decision_boundary_trainer.py configs/decision_boundary_config.yaml
```

## Data Generation

### Torus Dataset Generation

#### `src/data/dataset.py`

Main dataset generation module for torus pairs.

```python
from src.data.dataset import generate

# Generate linked/unlinked torus pairs
X, y = generate(
    n=1000,              # Points per torus
    link_exists=True,    # Whether tori are linked
    return_cloud=True    # Return point cloud (not distance matrix)
)
```

#### `src/data/gen_easy.py`

Simplified torus generation for testing.

```python
from src.data.gen_easy import gen_easy

# Generate simple torus dataset
X, y = gen_easy(
    nT=[500, 500],       # Points per torus
    Rs=[10, 10],         # Major radii
    rs=[3, 3],           # Minor radii
    link_exists=[0, 1]   # 0=unlinked, 1=linked
)
```

### Data Visualization

```bash
# 2D visualization
python src/visualization/visualize_dataset.py

# 3D interactive visualization
python src/visualization/visualize_dataset_3d.py
```

## Model Training

### Standard PyTorch Training

#### `src/models/torch_mlp.py`

Modern MLP implementation with activation extraction.

```bash
# Train MLP and extract activations
python src/models/torch_mlp.py configs/training_config.yaml
```

```python
# Programmatic usage
from src.models.torch_mlp import SimpleMLP, train_model

model = SimpleMLP(
    input_dim=3,
    num_hidden_layers=8,
    hidden_dim=32,
    output_dim=1,
    activation_fn_name='relu',
    output_activation_fn_name='sigmoid'
)

train_model("configs/training_config.yaml")
```

#### `src/models/torch_custom.py`

Flexible architecture supporting mixed layer types (Conv, Linear).

```bash
# Train custom architecture
python src/models/torch_custom.py configs/training_config.yaml
```

### Parallel Training

#### `src/models/torch_parallel.py`

Train multiple networks concurrently.

```bash
# Train 10 networks in parallel
python src/models/torch_parallel.py configs/training_config.yaml
```

Configuration in `training_config.yaml`:
```yaml
training:
  num_networks: 10
  max_parallel_workers: 4
```

### Vectorized Training

#### `src/models/torch_vectorized.py`

Efficient simultaneous training using PyTorch's vmap.

```bash
# Train multiple networks vectorized
python src/models/torch_vectorized.py configs/training_config.yaml
```

### Apple Silicon MLX Training

```bash
# MLX implementation (M1/M2/M3 optimized)
python src/models/trainer_mlx.py configs/training_config.yaml

# Parallel MLX training
python src/models/trainer_mlx_parallel.py configs/training_config.yaml
```

## Topological Analysis

### Persistent Homology Computation

#### `src/topology/compute_homology.py`

Main homology computation on layer activations.

```bash
# Compute homology for saved layer outputs
python src/topology/compute_homology.py
```

```python
# Programmatic usage
from src.topology.compute_homology import compute_layer_homology

compute_layer_homology(
    config_path="configs/homology_config.yaml"
)
```

#### Alternative Implementations

```bash
# Ripser-based (often faster)
python src/topology/compute_homology_ripser.py

# Witness complex approach
python src/topology/compute_homology_witness.py

# Decision boundary homology
python src/topology/compute_boundary_homology.py
```

### Ground Truth Homology

```bash
# Compute theoretical homology of torus
python src/utils/compute_torus_homology_original.py

# Ripser version
python src/utils/compute_torus_homology_ripser.py
```

## Network Graph Homology

### Factor Graph Approach for CNNs

The implementation uses factor graphs to handle weight sharing in convolutional layers:

```
Input Activations → Parameter Nodes → Output Activations
```

### Network Homology Tracker

```python
from src.topology.network_homology_tracker import NetworkHomologyTracker

# Initialize tracker
tracker = NetworkHomologyTracker(config)

# Track during training
distance, snapshot = tracker.track_training_step(
    model=model,
    step=100,
    epoch=5,
    validation_accuracy=0.92
)

# Get correlation with validation
correlation = tracker.compute_correlation_with_validation()
```

### Graph Construction

```python
from src.utils.network_graph_builder import UnifiedGraphBuilder

# Build network graph
builder = UnifiedGraphBuilder()
graph = builder.build_network_graph(model)
```

### Distance Metrics

```python
from src.analysis.persistence_distances import PersistenceDistanceCalculator

calculator = PersistenceDistanceCalculator()

# Compute various distances
wasserstein = calculator.wasserstein_distance(diagram1, diagram2)
heat_kernel = calculator.heat_kernel_distance(diagram1, diagram2, sigma=0.1)
bottleneck = calculator.bottleneck_distance(diagram1, diagram2)
```

## Visualization

### Betti Curves

```bash
# Basic Betti curve visualization
python src/visualization/plot_curves.py

# Advanced statistical analysis
python src/visualization/betti_curves.py \
    --input-file results/homology/layer_betti_numbers.pt \
    --output-dir results/plots
```

### Decision Boundaries

```bash
# 3D boundary visualization
python src/visualization/decision_boundary_viz.py \
    --epoch 50 \
    --azimuth 30 \
    --elevation 20
```

### Dataset Visualization

```bash
# 2D projections
python src/visualization/visualize_dataset.py

# 3D interactive
python src/visualization/visualize_dataset_3d.py \
    --n_points 1000 \
    --show_labels
```

### UMAP Visualization

```bash
# Dimensionality reduction plots
python src/visualization/uma_plot.py
```

## Configuration Files

### `configs/training_config.yaml`

Controls neural network architecture and training parameters.

```yaml
model:
  input_dim: 3              # Input dimension (3 for 3D torus data)
  num_hidden_layers: 8      # Number of hidden layers
  hidden_dim: 32            # Neurons per hidden layer
  output_dim: 1             # Output dimension (1 for binary classification)
  activation_fn_name: 'relu'
  output_activation_fn_name: 'sigmoid'
  dropout_rate: 0.0012      # Dropout probability
  use_batch_norm: false     # Batch normalization

training:
  device: 'cpu'             # 'cpu', 'cuda', 'mps', or 'auto'
  epochs: 100               # Training epochs
  batch_size: 64            # Batch size
  learning_rate: 0.001      # Initial learning rate
  seed: 42                  # Random seed
  
  optimizer:
    name: 'adamw'           # 'adam', 'adamw', 'sgd'
    weight_decay: 0.0001    # L2 regularization
  
  regularization:
    l1_lambda: 0.0          # L1 regularization strength
    l2_lambda: 0.0          # L2 regularization strength
  
  lr_scheduler:
    type: 'reduce_on_plateau'  # Learning rate scheduling
    factor: 0.1             # Reduction factor
    patience: 10            # Epochs before reduction
  
  early_stopping:
    enabled: true
    patience: 20            # Epochs without improvement
    min_delta: 0.0001       # Minimum improvement

data:
  type: 'synthetic'         # Data type
  synthetic_type: 'torus'   # 'torus' or 'moons'
  
  generation:
    n: 1000                 # Points per torus
    big_radius: 3           # Major radius
    small_radius: 1         # Minor radius
    solid: true             # Solid or hollow torus
    interior_noise: 0.01    # Noise for solid torus
  
  split_ratio: 0.8          # Train/test split

layer_extraction:
  enabled: true             # Extract layer activations
  output_dir: 'results/layer_outputs'
  variable_length_output: true  # Preserve natural dimensions

# Custom architecture (for torch_custom.py)
custom_architecture:
  enabled: true
  input_shape: [3]          # Input shape
  layers:                   # Layer definitions
    - type: linear
      out_features: 32
      activation: relu
    - type: reshape
      shape: [16, 2]
    - type: conv1d
      out_channels: 32
      kernel_size: 2
      activation: relu
```

### `configs/homology_config.yaml`

Controls persistent homology computation.

```yaml
io:
  input_dir: "results/layer_outputs"     # Layer activation files
  output_dir: "results/homology"         # Output directory
  save_intermediate: false               # Save intermediate results

sampling:
  use_fps: true                          # Furthest Point Sampling
  fps_num_points: 12000                  # Target points after FPS
  min_points_threshold: 50               # Minimum points required
  adaptive_sampling: true                # Adaptive sample size
  max_sample_ratio: 0.01                 # Max fraction to sample

distance:
  k_neighbors: 35                        # k-NN graph neighbors
  metric: "euclidean"                    # Distance metric
  geodesic: true                         # Use geodesic distances

computation:
  max_dimension: 1                       # Max homology dimension
  max_edge_length: 3                     # Max edge in Rips complex
  normalize_data: false                  # Normalize activations
  collapse_edges: false                  # Edge collapse optimization

witness_complex:
  n_landmarks: 1000                      # Number of landmarks
  landmark_selection: "random"           # 'maxmin', 'fps', 'random'
  max_alpha_square: 2                    # Max filtration value
  relaxation: 1                          # Relaxation parameter

parallel:
  enabled: true                          # Enable parallelization
  num_workers: 4                         # Worker processes
  chunk_size: 1                          # Tasks per worker

output:
  save_diagrams: true                    # Save persistence diagrams
  save_betti: true                       # Save Betti numbers
  output_format: "pytorch"               # 'pytorch' or 'numpy'
```

### `configs/network_homology_config.yaml`

Controls network graph homology analysis.

```yaml
network_homology:
  enabled: true
  track_interval: 10                     # Track every N steps
  
  graph_construction:
    backend: "graph-tool"                # Graph library
    normalize_weights: true              # Normalize edge weights
    weight_threshold: 1e-6               # Minimum weight
    handle_negative_weights: true        # Reverse edges for negatives
    
    cnn:
      use_factor_graph: true             # Factor graph for conv layers
      include_spatial_structure: true    # Store spatial info
  
  simplicial_complex:
    max_dimension: 2                     # Max homology dimension
    max_edge_length: 1.0                 # Max edge weight
    backend: "flagser"                   # 'flagser' or 'gudhi'
  
  distance_metrics:
    primary_metric: "heat"               # Distance metric
    heat_sigma: 0.1                      # Heat kernel bandwidth
    wasserstein_p: 2                     # Wasserstein parameter
  
  visualization:
    plot_interval: 50                    # Plot every N steps
    plot_types:
      - "distance_vs_validation"
      - "persistence_evolution"
      - "betti_evolution"
```

### `configs/decision_boundary_config.yaml`

Controls decision boundary extraction.

```yaml
model:
  architecture: 'mlp'                    # Model type
  width: 100                             # Hidden layer width
  depth: 5                               # Number of layers
  activation: 'relu'                     # Activation function

training:
  epochs: 200                            # Training epochs
  batch_size: 32                         # Batch size
  learning_rate: 0.001                   # Learning rate
  device: 'cuda'                         # Compute device
  
  extraction_schedule:
    enabled: true                        # Enable extraction
    frequency: 5                         # Extract every N epochs
    epochs: [0, 10, 50, 100, 199]       # Specific epochs

data:
  dataset_type: 'synthetic_3d'           # Dataset type
  num_samples: 5000                      # Total samples
  noise_level: 0.1                       # Noise level
  train_test_split: 0.8                  # Train/test ratio

extraction:
  method: 'grid_sampling'                # Extraction method
  grid:
    bounds: [[-15, 15], [-15, 15], [-15, 15]]  # Grid bounds
    resolution: [100, 100, 100]          # Grid resolution
  
  boundary_detection:
    method: 'probability_based'          # Detection method
    threshold: 0.5                       # Decision threshold
    tolerance: 0.01                      # Boundary tolerance
  
  optimization:
    use_cache: true                      # Cache predictions
    batch_size: 10000                    # Batch for predictions
    parallel_workers: 4                  # Parallel workers

visualization:
  plot_epochs: [0, 50, 100, 199]        # Epochs to plot
  save_plots: true                       # Save plots
  show_plots: false                      # Display plots
  plot_format: 'png'                     # Output format
  dpi: 150                               # Resolution
  
  plot_types:
    - '3d_scatter'                       # 3D scatter plot
    - 'cross_sections'                   # 2D cross-sections
    - 'evolution_animation'              # Animation

output:
  base_dir: 'results/decision_boundaries'
  save_boundaries: true                  # Save boundary data
  save_models: true                      # Save model checkpoints
  save_metrics: true                     # Save training metrics
```

## Utilities and Tools

### Distance Computation

```python
# k-NN geodesic distance
from src.utils.distance_computation import knn_geodesic_distance

distance_matrix = knn_geodesic_distance(
    data,                # Point cloud
    k=50,                # Number of neighbors
    n_jobs=-1            # Use all cores
)
```

### Graph Utilities

```python
from src.utils.graph import distance, compute_distance_matrix

# Compute distance matrix using graph-tool
dist_matrix = distance(X, k=30)
```

### Parameter Optimization

```bash
# Grid search for hyperparameters
python src/utils/parameter_grid_search.py \
    --param learning_rate --values 0.001 0.01 0.1 \
    --param hidden_dim --values 32 64 128
```

### Batch Processing

```bash
# Run multiple experiments
python -m src.utils.batch_runner \
    --config configs/batch_experiments.yaml \
    --num_runs 10
```

## Testing

### Run All Tests

```bash
# Using test script (handles environment)
./run_tests.sh

# Direct pytest
pytest tests/ -v
```

### Specific Test Modules

```bash
# Pipeline tests
pytest tests/test_pipeline.py -v

# Homology tests
pytest tests/test_homology.py -v

# Network homology tests
python test_network_homology.py

# Boundary extraction test
python test_boundary_training.py
```

### Performance Tests

```bash
# Benchmark homology computation
python tests/benchmark_homology.py

# Profile memory usage
python -m memory_profiler tests/test_large_network.py
```

## Advanced Features

### Custom Data Loaders

```python
from src.models.torch_mlp import load_data_from_file

# Load custom dataset
X, y = load_data_from_file("path/to/dataset.pt")
```

### Checkpoint Management

```python
# Save checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
}, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Multi-GPU Training

```yaml
# In training_config.yaml
training:
  device: 'cuda'
  data_parallel: true
  gpu_ids: [0, 1, 2, 3]
```

### Experiment Tracking

```python
# Use with Weights & Biases
import wandb

wandb.init(project="homology-analysis")
wandb.config.update(config)
wandb.log({"loss": loss, "accuracy": accuracy})
```

### Custom Metrics

```python
from src.analysis.metrics import compute_topological_complexity

complexity = compute_topological_complexity(
    persistence_diagrams,
    method="total_persistence"
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

2. **CUDA Out of Memory**
   - Reduce batch_size in config
   - Reduce grid resolution for boundaries
   - Use gradient accumulation

3. **Slow Homology Computation**
   - Enable FPS sampling
   - Reduce max_dimension
   - Use Ripser instead of Gudhi

4. **Graph-tool Installation**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-graph-tool
   
   # macOS
   brew install graph-tool
   
   # Fallback to igraph
   pip install python-igraph
   ```

### Performance Optimization

1. **Parallel Processing**
   - Set appropriate num_workers
   - Use torch_vectorized for multiple networks
   - Enable parallel in homology_config.yaml

2. **Memory Management**
   - Use sparse matrices for large datasets
   - Enable caching in decision boundary extraction
   - Reduce fps_num_points for sampling

3. **GPU Utilization**
   - Ensure CUDA/MPS is properly configured
   - Use mixed precision training
   - Monitor GPU memory with nvidia-smi

## Output Structure

```
results/
├── models/                    # Trained models
│   ├── model_epoch_100.pt
│   └── best_model.pt
├── layer_outputs/             # Extracted activations
│   ├── torch_mlp_layer_outputs.pt
│   └── torch_custom_layer_outputs_varlen.pt
├── homology/                  # Topological analysis
│   ├── layer_betti_numbers.pt
│   ├── persistence_diagrams/
│   └── homology_computation.log
├── plots/                     # Visualizations
│   ├── dataset_3d.html
│   ├── betti_curves.png
│   └── decision_boundary_evolution.gif
├── decision_boundaries/       # Boundary data
│   ├── boundary_epoch_*.npz
│   └── evolution_summary.json
└── network_homology/          # Network graph analysis
    ├── homology_history.pkl
    ├── homology_evolution.csv
    └── correlation_plots/
```

## Best Practices

1. **Always activate conda environment**
   ```bash
   conda activate myenv
   ```

2. **Use configuration files** instead of hardcoding parameters

3. **Monitor resource usage** during training and homology computation

4. **Save intermediate results** for long computations

5. **Version control configs** along with code

6. **Document experiments** with clear naming conventions

7. **Test on small datasets** before full runs

8. **Use appropriate backends**:
   - Ripser for speed
   - Gudhi for features
   - Witness complex for very large datasets

This completes the comprehensive usage guide for the Homology repository.