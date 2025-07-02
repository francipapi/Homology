# Network Graph Homology Implementation

This document describes the implementation of persistent homology tracking for neural network architectures as graphs, following the methodology from "Persistent Homology Captures the Generalization of Neural Networks Without A Validation Set" (Pérez-Fernández et al., NeurIPS 2021).

## Overview

The implementation treats neural networks as weighted directed graphs and uses persistent homology to track topological changes during training. The key innovation is the **factor graph approach** for convolutional layers, which elegantly handles weight sharing.

## Key Components

### 1. Network Graph Builder (`src/utils/network_graph_builder.py`)

Constructs graph representations of neural networks:

- **MLPGraphBuilder**: Direct edges between neurons for fully connected layers
- **ConvGraphBuilder**: Factor graph approach for convolutional layers
- **UnifiedGraphBuilder**: Combines different layer types into a single graph

#### Factor Graph Approach for Conv Layers

The factor graph introduces parameter nodes to handle weight sharing:

```
Input Activations → Parameter Nodes → Output Activations
    (i,j,c_in)         W_ab            (i',j',c_out)
```

**Benefits:**
- Each kernel weight appears exactly once as a parameter node
- Scales with parameter count, not spatial dimensions
- Compatible with existing persistent homology pipeline

**Example:** For Conv2D(3, 64, kernel_size=3) on 32×32 input:
- Input nodes: 32×32×3 = 3,072
- Parameter nodes: 3×3×3×64 = 1,728 (only these carry weights!)
- Output nodes: 30×30×64 = 57,600
- Weight-carrying edges: 1,728 (self-loops on parameter nodes)

### 2. Simplicial Complex Construction (`src/utils/network_simplicial_complex.py`)

- **DirectedFlagComplex**: Handles directed graphs from neural networks
- **WeightedFiltration**: Implements weight-based filtration
- Supports multiple backends: Flagser (preferred), Gudhi (fallback)

### 3. Network Homology Tracker (`src/topology/network_homology_tracker.py`)

Main orchestration class that:
- Builds network graphs at specified intervals
- Computes persistent homology
- Tracks evolution during training
- Calculates distances between consecutive states
- Correlates with validation accuracy

### 4. Persistence Distance Metrics (`src/analysis/persistence_distances.py`)

Implements various distance metrics:
- **Wasserstein distance**: Optimal transport between diagrams
- **Bottleneck distance**: Maximum matching distance
- **Heat kernel distance**: Used in the paper, stable summary
- **Silhouette distance**: Functional summary approach

### 5. Training Integration (`src/models/trainer_with_homology.py`)

Extended trainer that incorporates homology tracking:
- Tracks homology at configurable intervals
- Computes correlation with validation metrics
- Saves results and visualizations

## Usage

### Basic Example

```python
from src.topology.network_homology_tracker import NetworkHomologyTracker
from src.models.trainer_with_homology import TrainerWithHomology

# Create model
model = YourNeuralNetwork()

# Load configurations
training_config = load_config('configs/training_config.yaml')
homology_config = load_config('configs/network_homology_config.yaml')

# Create trainer with homology tracking
trainer = TrainerWithHomology(model, training_config, homology_config)

# Train and track homology
results = trainer.train()

# Results include homology-validation correlation
print(f"Correlation: {results['homology_validation_correlation']}")
```

### Command Line Usage

```bash
# Train with homology tracking
python src/models/trainer_with_homology.py \
    --training-config configs/training_config.yaml \
    --homology-config configs/network_homology_config.yaml \
    --model-type custom
```

### Configuration

Key parameters in `configs/network_homology_config.yaml`:

```yaml
network_homology:
  track_interval: 10  # Track every N steps
  
  graph_construction:
    cnn:
      use_factor_graph: true  # Enable factor graphs
      
  distance_metrics:
    primary_metric: "heat"  # As used in paper
    heat_sigma: 0.1
```

## Implementation Details

### Graph Construction

1. **MLP Layers**: Direct edges from input to output neurons
   - Edge weight = connection weight
   - Negative weights reverse edge direction

2. **Conv Layers (Factor Graph)**:
   - Three node types: input, parameter, output
   - Self-loops on parameter nodes carry weight magnitudes
   - Structural edges (weight 1.0) encode connectivity
   - Sign-aware: negative weights reverse structural edges

3. **Other Layers**:
   - BatchNorm: Identity edges with learned scaling
   - Pooling: Edges with weight 1.0
   - Skip connections: Direct edges preserving residual structure

### Persistence Computation

1. Build directed flag complex from network graph
2. Use edge weights as filtration values
3. Compute persistent homology up to dimension 2
4. Extract Betti numbers and persistence diagrams

### Distance Tracking

The heat kernel distance (as in the paper) provides a stable metric:
- Computes heat kernel signature for each diagram
- Measures L2 distance between signatures
- Robust to small perturbations

## Testing

Run the test suite:

```bash
python test_network_homology.py
```

Tests include:
- MLP graph construction
- Conv factor graph construction
- Homology computation
- Distance metrics
- Complete pipeline integration

## Performance Considerations

### Computational Complexity
- Graph construction: O(parameters)
- Persistence: O(n³) worst case, typically O(n²)
- Distance: O(n×m) for diagrams of size n,m

### Optimization Strategies
- Cache graph structure (only weights change)
- Sparse representations for large networks
- Parallel distance computations
- Approximate methods for very large models

### Memory Requirements
- Small networks (<1M params): ~100MB overhead
- Medium networks (1-10M params): ~1GB overhead
- Large networks (>10M params): ~5GB overhead

## Expected Results

Based on the paper's findings:
- Heat distance correlation with validation: ~0.82
- Distance stabilization coincides with validation plateau
- Early layers show less topological change than later layers

## References

- Pérez-Fernández et al. "Persistent Homology Captures the Generalization of Neural Networks Without A Validation Set" (NeurIPS 2021)
- Flagser: Fast computation of directed flag complexes
- Gudhi: Computational topology and persistent homology
- Graph-tool: Efficient graph analysis library