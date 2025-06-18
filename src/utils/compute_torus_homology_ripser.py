"""
Compute homology of the torus training dataset using Ripser.

This script generates the torus dataset using the same parameters as the training
pipeline and computes persistent homology using Ripser with optimized settings
from homology_config.yaml.
"""

import numpy as np
import yaml
import time
import os
import sys
from pathlib import Path
from ripser import ripser
from typing import Dict, Tuple, List

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.dataset import generate
from src.utils.distance_computation import knn_geodesic_distance, farthest_point_sampling_pytorch


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
    """Apply FPS sampling if configured."""
    sampling_config = homology_config['sampling']
    
    if not sampling_config['use_fps']:
        return X
    
    target_points = sampling_config['fps_num_points']
    if X.shape[0] <= target_points:
        print(f"Dataset size ({X.shape[0]}) <= target FPS points ({target_points}), skipping sampling")
        return X
    
    print(f"Applying FPS sampling: {X.shape[0]} -> {target_points} points")
    start_time = time.time()
    
    # Use efficient PyTorch-based FPS
    X_sampled = farthest_point_sampling_pytorch(X, device='auto')
    
    sampling_time = time.time() - start_time
    print(f"FPS sampling completed in {sampling_time:.2f}s")
    
    return X_sampled


def check_graph_connectivity(dist_matrix: np.ndarray, max_edge_length: float) -> None:
    """Check if the graph is connected at the given threshold."""
    # Count infinite distances (disconnected components)
    infinite_count = np.sum(np.isinf(dist_matrix))
    total_entries = dist_matrix.size
    
    # Count reachable pairs within threshold
    reachable_pairs = np.sum((dist_matrix <= max_edge_length) & np.isfinite(dist_matrix))
    
    print(f"Graph connectivity analysis:")
    print(f"  Infinite distances: {infinite_count}/{total_entries} ({100*infinite_count/total_entries:.1f}%)")
    print(f"  Reachable pairs within threshold {max_edge_length}: {reachable_pairs}")
    
    # Check connectivity of components
    finite_mask = np.isfinite(dist_matrix)
    if np.all(finite_mask):
        print(f"  Graph appears fully connected (no infinite distances)")
    else:
        print(f"  Graph has disconnected components")


def compute_distance_matrix(X: np.ndarray, homology_config: Dict) -> np.ndarray:
    """Compute distance matrix using configuration settings."""
    distance_config = homology_config['distance']
    comp_config = homology_config['computation']
    
    if distance_config['geodesic']:
        print(f"Computing geodesic distances on k-NN graph (k={distance_config['k_neighbors']})")
        start_time = time.time()
        
        # Use the optimized knn_geodesic_distance function
        dist_matrix = knn_geodesic_distance(X)
        
        distance_time = time.time() - start_time
        print(f"Distance matrix computed in {distance_time:.2f}s, shape: {dist_matrix.shape}")
        
        # Check connectivity
        check_graph_connectivity(dist_matrix, comp_config['max_edge_length'])
        
        return dist_matrix
    else:
        print("Computing Euclidean distance matrix")
        start_time = time.time()
        
        # Compute pairwise Euclidean distances
        diff = X[:, None, :] - X[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        
        distance_time = time.time() - start_time
        print(f"Distance matrix computed in {distance_time:.2f}s, shape: {dist_matrix.shape}")
        return dist_matrix


def compute_homology_ripser(dist_matrix: np.ndarray, homology_config: Dict) -> Dict:
    """Compute persistent homology using Ripser."""
    comp_config = homology_config['computation']
    
    print(f"Computing persistent homology with Ripser:")
    print(f"  Max dimension: {comp_config['max_dimension']}")
    print(f"  Max edge length: {comp_config['max_edge_length']}")
    print(f"  Distance matrix shape: {dist_matrix.shape}")
    
    start_time = time.time()
    
    # Run Ripser with correct parameters for distance matrix
    result = ripser(
        dist_matrix,
        maxdim=comp_config['max_dimension'],
        thresh=comp_config['max_edge_length'],  
        distance_matrix=True,  # Explicitly specify this is a distance matrix
        do_cocycles=False  # Don't compute cocycles for efficiency
    )
    
    computation_time = time.time() - start_time
    print(f"Ripser computation completed in {computation_time:.2f}s")
    
    return result


def extract_betti_numbers(ripser_result: Dict, homology_config: Dict) -> List[int]:
    """Extract Betti numbers from Ripser result."""
    comp_config = homology_config['computation']
    max_dim = comp_config['max_dimension']
    min_persistence = comp_config['min_persistence']
    max_edge_length = comp_config['max_edge_length']
    
    betti_numbers = []
    
    print("Analyzing persistence diagrams:")
    
    for dim in range(max_dim + 1):
        if dim < len(ripser_result['dgms']):
            diagram = ripser_result['dgms'][dim]
            print(f"  H{dim}: {len(diagram)} total intervals")
            
            if len(diagram) == 0:
                betti_numbers.append(0)
                continue
            
            # Filter by minimum persistence
            if min_persistence > 0:
                persistence = diagram[:, 1] - diagram[:, 0]
                valid_features = diagram[persistence >= min_persistence]
            else:
                valid_features = diagram
            
            if dim == 0:
                # For H0: count infinite intervals (connected components)
                infinite_features = valid_features[np.isinf(valid_features[:, 1])]
                finite_features = valid_features[np.isfinite(valid_features[:, 1])]
                betti = len(infinite_features)
                print(f"  H{dim}: {len(infinite_features)} infinite intervals (connected components)")
                print(f"  H{dim}: {len(finite_features)} finite intervals")
                if len(finite_features) > 0:
                    print(f"  H{dim}: finite death times range: [{finite_features[:, 1].min():.3f}, {finite_features[:, 1].max():.3f}]")
            else:
                # For H1, H2, etc.: count intervals that persist to max_edge_length
                # or are still "alive" (close to max threshold)
                
                # Method 1: Count features that persist significantly
                persistence_values = valid_features[:, 1] - valid_features[:, 0]
                significant_features = valid_features[persistence_values >= min_persistence]
                
                # Method 2: Count features still alive near the threshold
                near_threshold = valid_features[
                    (valid_features[:, 1] >= max_edge_length * 0.8) | 
                    np.isinf(valid_features[:, 1])
                ]
                
                # Show persistence range for debugging
                if len(valid_features) > 0:
                    birth_range = f"[{valid_features[:, 0].min():.3f}, {valid_features[:, 0].max():.3f}]"
                    death_range = f"[{valid_features[np.isfinite(valid_features[:, 1]), 1].min():.3f}, {valid_features[np.isfinite(valid_features[:, 1]), 1].max():.3f}]" if np.any(np.isfinite(valid_features[:, 1])) else "[no finite deaths]"
                    print(f"  H{dim}: birth range: {birth_range}, death range: {death_range}")
                
                # Use the more conservative count
                betti = len(near_threshold)
                print(f"  H{dim}: {len(significant_features)} significant features, {len(near_threshold)} persistent to threshold")
                
            betti_numbers.append(betti)
        else:
            betti_numbers.append(0)
    
    return betti_numbers


def save_results(ripser_result: Dict, betti_numbers: List[int], X: np.ndarray, 
                homology_config: Dict) -> None:
    """Save homology computation results."""
    output_config = homology_config['output']
    
    if not output_config['save_betti'] and not output_config['save_diagrams']:
        return
    
    # Create output directory
    output_dir = Path("results/torus_homology")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_config['save_betti']:
        betti_path = output_dir / "torus_betti_numbers.npy"
        np.save(betti_path, np.array(betti_numbers))
        print(f"Betti numbers saved to: {betti_path}")
    
    if output_config['save_diagrams']:
        # Save persistence diagrams
        for dim, diagram in enumerate(ripser_result['dgms']):
            if len(diagram) > 0:
                diag_path = output_dir / f"torus_persistence_diagram_dim_{dim}.npy"
                np.save(diag_path, diagram)
                print(f"H{dim} persistence diagram saved to: {diag_path}")
    
    # Save dataset info
    info = {
        'dataset_shape': X.shape,
        'betti_numbers': betti_numbers,
        'max_dimension': len(betti_numbers) - 1,
        'computation_method': 'ripser'
    }
    
    info_path = output_dir / "torus_homology_info.yaml"
    with open(info_path, 'w') as f:
        yaml.dump(info, f, default_flow_style=False)
    print(f"Computation info saved to: {info_path}")


def main():
    """Main function to compute torus homology using Ripser."""
    print("=" * 60)
    print("Computing Homology of Torus Training Dataset (Ripser)")
    print("=" * 60)
    
    # Load configurations
    training_config, homology_config = load_configs()
    
    # Generate dataset
    X, y = generate_torus_dataset(training_config)
    
    # Apply sampling if configured
    X_processed = apply_sampling(X, homology_config)
    
    # Compute distance matrix
    dist_matrix = compute_distance_matrix(X_processed, homology_config)
    
    # Compute persistent homology
    ripser_result = compute_homology_ripser(dist_matrix, homology_config)
    
    # Extract Betti numbers
    betti_numbers = extract_betti_numbers(ripser_result, homology_config)
    
    # Display results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"Dataset size: {X.shape[0]} -> {X_processed.shape[0]} points")
    print(f"Betti numbers: {betti_numbers}")
    for i, betti in enumerate(betti_numbers):
        print(f"  H{i}: {betti}")
    
    # Save results
    save_results(ripser_result, betti_numbers, X_processed, homology_config)
    
    print("\nTorus homology computation completed!")


if __name__ == "__main__":
    main()