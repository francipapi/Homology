"""
Compute homology of the torus training dataset using Gudhi with Ripser-style optimizations.

This script implements Gudhi persistent homology computation following Ripser best practices:
- Efficient distance matrix computation with geodesic distances
- Optimized memory usage through sparse representations
- Fast persistence computation using Gudhi's RipsComplex
- Parallel processing where applicable
- Direct persistence diagram computation without full complex construction
"""

import numpy as np
import yaml
import time
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.dataset import generate
from src.utils.distance_computation import knn_geodesic_distance, farthest_point_sampling_pytorch

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: Gudhi not available. Please install with: pip install gudhi")


def load_configs() -> Tuple[Dict, Dict]:
    """Load training and homology configurations."""
    training_config_path = Path(__file__).parent.parent.parent / "configs" / "training_config.yaml"
    homology_config_path = Path(__file__).parent.parent.parent / "configs" / "homology_config.yaml"
    
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    with open(homology_config_path, 'r') as f:
        homology_config = yaml.safe_load(f)
    
    return training_config, homology_config


def generate_torus_dataset(training_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Generate torus dataset using training configuration parameters."""
    print("Generating torus dataset...")
    data_params = training_config['data']['generation']
    
    X, y = generate(
        n=data_params['n'],
        big_radius=data_params['big_radius'],
        small_radius=data_params['small_radius'],
        solid=data_params.get('solid', False),
        interior_noise=data_params.get('interior_noise', 0.1)
    )
    
    print(f"Generated dataset: {X.shape[0]} points, {X.shape[1]} dimensions")
    return X, y


def apply_sampling(X: np.ndarray, homology_config: Dict) -> np.ndarray:
    """Apply FPS sampling using efficient implementation."""
    sampling_config = homology_config['sampling']
    
    if not sampling_config['use_fps']:
        return X
    
    target_points = sampling_config['fps_num_points']
    if X.shape[0] <= target_points:
        print(f"Dataset size ({X.shape[0]}) <= target FPS points ({target_points}), skipping sampling")
        return X
    
    print(f"Applying FPS sampling: {X.shape[0]} -> {target_points} points")
    start_time = time.time()
    
    # Use efficient PyTorch-based FPS (same as Ripser implementation)
    X_sampled = farthest_point_sampling_pytorch(X, device='auto')
    
    sampling_time = time.time() - start_time
    print(f"FPS sampling completed in {sampling_time:.2f}s")
    
    return X_sampled


def compute_distance_matrix(X: np.ndarray, homology_config: Dict) -> np.ndarray:
    """Compute distance matrix using configuration settings."""
    distance_config = homology_config['distance']
    
    if distance_config['geodesic']:
        print(f"Computing geodesic distances on k-NN graph (k={distance_config['k_neighbors']})...")
        start_time = time.time()
        
        dist_matrix = knn_geodesic_distance(X)
        
        distance_time = time.time() - start_time
        print(f"Geodesic distance computation completed in {distance_time:.2f}s")
        
    else:
        print("Using Euclidean distances (no explicit distance matrix needed)")
        dist_matrix = None
    
    return dist_matrix


def compute_persistence_efficient(X: np.ndarray, homology_config: Dict, 
                                 distance_matrix: Optional[np.ndarray] = None) -> Tuple[List, gudhi.SimplexTree, float]:
    """
    Compute persistence using Gudhi's most efficient methods with proper dimension handling.
    
    This follows Ripser-style optimizations and Gudhi documentation best practices:
    1. Use sparse representations where possible
    2. Minimize memory usage
    3. Proper dimension handling after operations
    4. Use optimized algorithms for persistence computation
    """
    comp_config = homology_config['computation']
    
    print(f"\nComputing persistent homology (Gudhi-Ripser style):")
    print(f"  Points: {X.shape[0]}")
    print(f"  Max edge length: {comp_config['max_edge_length']}")
    print(f"  Max dimension: {comp_config['max_dimension']}")
    
    start_time = time.time()
    
    # Use edge collapse for efficiency (key Ripser-style optimization)
    if distance_matrix is not None:
        print("  Using precomputed distance matrix")
        
        # Create Rips complex from distance matrix
        rips_complex = gudhi.RipsComplex(
            distance_matrix=distance_matrix,
            max_edge_length=comp_config['max_edge_length']
        )
    else:
        print("  Using Euclidean distances")
        
        # For Euclidean case, use sparse edge list for efficiency
        rips_complex = gudhi.RipsComplex(
            points=X,
            max_edge_length=comp_config['max_edge_length'],
            sparse=comp_config.get('sparse', 0.3)  # Ripser-style sparse optimization
        )
    
    # Create simplex tree with dimension restriction
    simplex_tree = rips_complex.create_simplex_tree(
        max_dimension=1
    )
    
    
    
    # Apply edge collapse for efficiency (Ripser-style optimization)
    if not comp_config.get('collapse_edges', False):  # Only if not disabled
        print("  Applying edge collapse optimization...")
        collapse_start = time.time()
        
        # Gudhi's edge collapse is similar to Ripser's optimization
        num_collapsed = simplex_tree.collapse_edges()
        print(f"    Collapsed {num_collapsed} edges")
        
        # Check dimension after collapse (following Gudhi documentation)
        post_collapse_dim = simplex_tree.dimension()
        print(f"    Dimension after collapse: {post_collapse_dim}")
        print(f"    Simplices after collapse: {simplex_tree.num_simplices()}")
        
        # IMPORTANT: Expand after collapse to recover full complex
        # This is the key insight - we need to re-expand to max dimension
        max_expansion_dim = comp_config['max_dimension'] + 1
        if post_collapse_dim < max_expansion_dim:
            print(f"  Expanding to dimension {max_expansion_dim}...")
            simplex_tree.expansion(max_expansion_dim)
            
            # Verify dimension after expansion (following Gudhi documentation)
            final_dimension = simplex_tree.dimension()
            print(f"    Final dimension after expansion: {final_dimension}")
            print(f"    Final simplices: {simplex_tree.num_simplices()}")
        else:
            print(f"  No expansion needed (current dim: {post_collapse_dim}, target: {max_expansion_dim})")
        
        collapse_time = time.time() - collapse_start
        print(f"  Edge collapse + expansion completed in {collapse_time:.2f}s")
    
    # Verify final complex dimension before persistence computation
    final_complex_dim = simplex_tree.dimension()
    print(f"  Final complex dimension: {final_complex_dim}")
    
    # Compute persistence with coefficient field optimization
    # Using field coefficient 2 (like Ripser) for faster computation
    persistence = simplex_tree.persistence(
        homology_coeff_field=2,  # Z/2Z coefficients (Ripser default)
        min_persistence=comp_config['min_persistence']
    )
    
    computation_time = time.time() - start_time
    
    return persistence, simplex_tree, computation_time


def extract_betti_numbers_builtin(simplex_tree: gudhi.SimplexTree, max_dim: int) -> List[int]:
    """Extract Betti numbers using Gudhi's built-in method."""
    try:
        # Use Gudhi's built-in betti_numbers() method
        betti_numbers = simplex_tree.betti_numbers()
        
        return betti_numbers[:max_dim + 2]
    except Exception as e:
        print(f"Warning: Gudhi betti_numbers() failed: {e}")
        return [0] * (max_dim + 1)


def save_results(betti_numbers: List[int], X: np.ndarray, 
                homology_config: Dict, computation_time: float, 
                distance_matrix_time: Optional[float] = None) -> None:
    """Save computation results."""
    output_config = homology_config['output']
    
    # Create output directory
    output_dir = Path("results/torus_homology")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_config['save_betti']:
        betti_path = output_dir / "torus_betti_numbers_gudhi.npy"
        np.save(betti_path, np.array(betti_numbers))
        print(f"Betti numbers saved to: {betti_path}")
    
    # Save computation info
    info = {
        'dataset_shape': X.shape,
        'betti_numbers': betti_numbers,
        'computation_method': 'gudhi_builtin',
        'computation_time': computation_time,
        'distance_matrix_time': distance_matrix_time
    }
    
    info_path = output_dir / "torus_homology_info_gudhi.yaml"
    with open(info_path, 'w') as f:
        yaml.dump(info, f, default_flow_style=False)
    print(f"Computation info saved to: {info_path}")


def main():
    """Main function implementing Gudhi computation with Ripser-style optimizations."""
    if not GUDHI_AVAILABLE:
        print("Error: Gudhi is required but not installed.")
        print("Please install with: pip install gudhi")
        return
    
    print("=" * 60)
    print("Computing Homology of Torus Dataset (Gudhi Built-in)")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # Load configurations
    training_config, homology_config = load_configs()
    
    # Generate dataset
    X, _ = generate_torus_dataset(training_config)
    
    # Apply sampling (using same efficient method as Ripser)
    X_processed = apply_sampling(X, homology_config)
    
    # Compute distance matrix if needed
    distance_matrix = None
    distance_matrix_time = None
    
    if homology_config['distance']['geodesic']:
        dist_start = time.time()
        distance_matrix = compute_distance_matrix(X_processed, homology_config)
        distance_matrix_time = time.time() - dist_start
    
    # Compute persistence using Ripser-style optimizations
    persistence, simplex_tree, comp_time = compute_persistence_efficient(
        X_processed, homology_config, distance_matrix
    )
    
    # Extract Betti numbers using Gudhi's built-in method
    max_edge_length = homology_config['computation']['max_edge_length']
    max_dim = homology_config['computation']['max_dimension']
    
    # Use Gudhi's built-in betti_numbers() method (most reliable)
    betti_numbers = extract_betti_numbers_builtin(simplex_tree, max_dim)
    print(f"\nBetti numbers (from Gudhi built-in method): {betti_numbers}")
    
    total_time = time.time() - total_start_time
    
    # Display results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"Dataset size: {X.shape[0]} -> {X_processed.shape[0]} points")
    print(f"Betti numbers: {betti_numbers}")
    for i, betti in enumerate(betti_numbers):
        print(f"  β{i} = {betti}")
    
    print(f"\nComputation time: {comp_time:.2f}s")
    if distance_matrix_time:
        print(f"Distance matrix time: {distance_matrix_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    
    # Save results
    save_results(betti_numbers, X_processed, homology_config, 
                comp_time, distance_matrix_time)
    
    print(f"\nGudhi computation completed!")


if __name__ == "__main__":
    main()