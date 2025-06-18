# Enhanced Maxmin Landmark Selection Implementation

## Overview

This document describes the enhanced maxmin landmark selection algorithm implemented for witness complex homology computation in `compute_homology_witness.py`.

## What is Maxmin Selection?

The maxmin algorithm is a greedy landmark selection strategy that iteratively selects points to maximize coverage of the data space. At each step, it chooses the point that is furthest from all previously selected landmarks, ensuring good geometric distribution.

## Key Improvements

### 1. **Smart Initialization Strategies**
- **Center**: Start with the point closest to the data centroid
- **Corner**: Start with the point furthest from the centroid (good for bounded datasets)
- **PCA**: Start with a point along the first principal component (experimental)
- **Random**: Traditional random initialization

### 2. **Memory-Efficient Batch Processing**
- Large datasets are processed in batches to manage memory usage
- Configurable batch size (default: 10,000 points)
- Maintains accuracy while reducing memory footprint

### 3. **Noise Injection for Tie-Breaking**
- Optional small noise addition to break ties in distance computations
- Improves determinism and coverage quality
- Configurable noise scale

### 4. **Both PyTorch and NumPy Implementations**
- **PyTorch version**: GPU-accelerated, used in direct homology computation
- **NumPy version**: CPU-optimized, used in main processing pipeline
- Consistent algorithms across both implementations

## Configuration Options

### Basic Landmark Configuration
```yaml
witness_complex:
  landmark_selection: "maxmin"     # "maxmin", "fps", or "random"
  n_landmarks: 400                 # Fixed number of landmarks
  adaptive_landmarks: true         # Enable adaptive sizing
  landmark_percentage: 0.005       # 0.5% of points when adaptive
  min_landmarks: 20               # Minimum landmarks
  max_landmarks: 500              # Maximum landmarks
```

### Maxmin Algorithm Configuration
```yaml
witness_complex:
  # Initialization strategy
  maxmin_init_strategy: "center"   # "center", "corner", "pca", or "random"
  
  # Tie-breaking (optional)
  maxmin_add_noise: false         # Add noise to break ties
  maxmin_noise_scale: 1e-6        # Noise scale
  
  # Performance tuning
  batch_size: 1000                # Batch size for distance computations
```

## Algorithm Details

### Maxmin Selection Process

1. **Initialization**: Select first landmark using chosen strategy
2. **Distance Computation**: Calculate distances from all points to the first landmark
3. **Iterative Selection**: For each remaining landmark:
   - Find the point with maximum distance to its nearest existing landmark
   - Update distance matrix with new landmark
   - Repeat until desired number of landmarks is reached

### Complexity
- **Time**: O(n * k * d) where n=points, k=landmarks, d=dimensions
- **Space**: O(n) for distance storage
- **Batch Processing**: Reduces peak memory from O(n²) to O(batch_size²)

## Performance Comparison

| Method | Coverage Quality | Speed | Memory Usage |
|--------|-----------------|-------|--------------|
| Random | Baseline | Fastest | Minimal |
| Maxmin (Random Init) | Better | Moderate | Moderate |
| Maxmin (Center Init) | Best | Moderate | Moderate |
| Maxmin (Corner Init) | Good | Moderate | Moderate |

## Usage Examples

### Direct Function Usage
```python
import torch
from topology.compute_homology_witness import select_landmarks_maxmin_torch

# Generate test data
points = torch.randn(1000, 3)  # 1000 points in 3D

# Select 50 landmarks using maxmin with center initialization
landmarks = select_landmarks_maxmin_torch(
    points, 
    n_landmarks=50,
    init_strategy='center',
    batch_size=5000
)
```

### Configuration-Based Usage
```python
from topology.compute_homology_witness import compute_layer_homology_witness_torch

# Set landmark selection to maxmin in config
config_path = "configs/homology_config.yaml"

# Run witness complex computation with maxmin landmarks
compute_layer_homology_witness_torch(config_path)
```

## Benefits for Topological Analysis

1. **Better Coverage**: Maxmin ensures landmarks are well-distributed across the data manifold
2. **Topology Preservation**: Good landmark placement preserves important topological features
3. **Consistency**: Deterministic selection (with fixed seeds) improves reproducibility
4. **Scalability**: Batch processing enables application to large datasets

## Initialization Strategy Guidelines

- **Center**: Best for most datasets, provides stable central starting point
- **Corner**: Good for datasets with clear boundaries or outliers
- **Random**: Use when other strategies fail or for comparison
- **PCA**: Experimental, may help with highly anisotropic data

## Implementation Files

- `src/topology/compute_homology_witness.py`: Main implementation
- `configs/homology_config.yaml`: Configuration settings
- `test_maxmin_landmarks.py`: Demonstration and comparison script
- `test_maxmin_config.py`: Configuration testing script

## Future Enhancements

1. **Adaptive Batch Sizing**: Automatically adjust batch size based on available memory
2. **Progressive Selection**: Use coarse-to-fine landmark refinement
3. **Clustering-Based Initialization**: Use k-means centers as initial landmarks
4. **GPU Memory Management**: Better CUDA memory handling for very large datasets

## References

1. Silva, V. de, & Carlsson, G. (2004). Topological estimation using witness complexes
2. Otter, N., et al. (2017). A roadmap for the computation of persistent homology
3. Dey, T. K., et al. (2008). Approximating cycles in a shortest basis of the first homology group from point data