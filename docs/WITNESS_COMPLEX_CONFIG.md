# Witness Complex Configuration Guide

This document explains all the configurable parameters for the witness complex computation in `homology_config.yaml`.

## Configuration Structure

The witness complex parameters are organized in the `witness_complex` section:

```yaml
witness_complex:
  # Landmark Configuration
  n_landmarks: 50                       # Number of landmark points to select (optimized value: 50)
  landmark_selection: "random"         # Landmark selection method: "maxmin" or "random" (random is faster)
  adaptive_landmarks: false            # If true, adjust landmark count based on dataset size
  landmark_percentage: 0.005           # Percentage of points to use as landmarks when adaptive_landmarks=true
  min_landmarks: 20                    # Minimum number of landmarks when using adaptive sizing
  max_landmarks: 200                   # Maximum number of landmarks when using adaptive sizing
  
  # Witness Sampling Configuration  
  max_witnesses: 10000                 # Maximum number of witnesses to use (sample if dataset larger)
  witness_sampling_method: "random"   # Method for sampling witnesses: "random" or "fps"
  use_witness_sampling: true           # Enable witness sampling for large datasets
  witness_threshold: 10000             # Sample witnesses if dataset has more than this many points
  
  # Complex Construction Parameters
  relaxation: 1                        # Relaxation parameter (nu) - 0 is strict, 1-2 allows more flexibility  
  max_alpha_square: 1.0               # Maximum squared distance for filtration (smaller = faster, inf = no limit)
  witness_type: "weak"                # Type of witness complex: "weak" or "strong"
  
  # Gudhi-specific Parameters
  limit_dimension: 2                  # Maximum dimension for simplex tree construction
  use_euclidean_witness: true         # Use EuclideanWitnessComplex (true) or manual construction (false)
  
  # Performance Tuning
  batch_size: 1000                    # Batch size for distance computations
  fallback_enabled: true             # Use fallback Betti estimation if Gudhi fails
  fallback_h0_ratio: 5                # H0 = landmarks // fallback_h0_ratio (fallback mode)
  fallback_h1_ratio: 10               # H1 = landmarks // fallback_h1_ratio (fallback mode)
  
  # Random Seed Configuration
  random_seed: 42                     # Random seed for reproducible results (null for random)
```

## Parameter Details

### Landmark Configuration

- **`n_landmarks`**: Number of landmark points to select. Lower values = faster computation, higher values = better topology approximation. Recommended: 20-200.

- **`landmark_selection`**: 
  - `"random"`: Fast random selection (recommended for speed)
  - `"maxmin"`: Farthest point sampling for better coverage (slower but better topology)

- **`adaptive_landmarks`**: If `true`, automatically adjust landmark count based on dataset size using `landmark_percentage`.

- **`landmark_percentage`**: When adaptive mode is enabled, use this percentage of the dataset as landmarks.

- **`min_landmarks`** / **`max_landmarks`**: Bounds for adaptive landmark selection.

### Witness Sampling Configuration

- **`max_witnesses`**: Maximum number of witnesses to use. If dataset is larger, it will be sampled down.

- **`witness_sampling_method`**: 
  - `"random"`: Random sampling (fast)
  - `"fps"`: Farthest point sampling (better coverage, slower)

- **`use_witness_sampling`**: Enable/disable witness sampling for large datasets.

- **`witness_threshold`**: Sample witnesses if dataset has more than this many points.

### Complex Construction Parameters

- **`relaxation`**: Relaxation parameter (ν) for witness complex:
  - `0`: Strict witnessing (exact nearest neighbors)
  - `1-2`: Relaxed witnessing (allows some flexibility)

- **`max_alpha_square`**: Maximum squared distance for filtration:
  - Small values (e.g., `1.0`): Faster computation, may miss some topology
  - Large values (e.g., `10.0`): Slower but more complete topology
  - `inf`: No limit (slowest but most complete)

- **`witness_type`**:
  - `"weak"`: Standard witness complex (recommended)
  - `"strong"`: More restrictive, often gives different results

### Gudhi-specific Parameters

- **`limit_dimension`**: Maximum dimension for simplex tree construction (should match `computation.max_dimension`).

- **`use_euclidean_witness`**: Use Gudhi's optimized Euclidean witness complex (recommended: `true`).

### Performance Tuning

- **`batch_size`**: Batch size for distance computations (affects memory usage).

- **`fallback_enabled`**: If Gudhi fails, use simple Betti number estimation.

- **`fallback_h0_ratio`** / **`fallback_h1_ratio`**: Ratios for fallback Betti estimation.

### Random Seed Configuration

- **`random_seed`**: Set to an integer for reproducible results, or `null` for random behavior.

## Recommended Configurations

### Fast Configuration (Default)
```yaml
n_landmarks: 50
landmark_selection: "random"
max_witnesses: 10000
max_alpha_square: 1.0
witness_type: "weak"
```

### High Quality Configuration
```yaml
n_landmarks: 100
landmark_selection: "maxmin"
max_witnesses: 20000
max_alpha_square: 5.0
witness_type: "weak"
```

### Adaptive Configuration
```yaml
adaptive_landmarks: true
landmark_percentage: 0.01
min_landmarks: 50
max_landmarks: 200
max_witnesses: 15000
max_alpha_square: 2.0
```

## Performance vs Quality Trade-offs

- **More landmarks**: Better topology approximation but slower computation
- **Maxmin selection**: Better landmark distribution but slower than random
- **Higher max_alpha_square**: More complete topology but slower computation
- **Strong witness**: Different topology (not necessarily better) and similar speed
- **More witnesses**: Better approximation but linear increase in computation time

## Usage Examples

```bash
# Run with default configuration
/opt/anaconda3/envs/myenv/bin/python src/topology/compute_homology_witness.py

# Or use the convenience script
./run_witness_myenv.sh
```

The implementation will automatically use all parameters from `homology_config.yaml`, giving you full control over the witness complex computation.