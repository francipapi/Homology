"""
Compute Wasserstein distance between persistence diagrams of train and test datasets
to quantify network generalization.

This script loads train and test layer activations, computes their persistence diagrams,
and calculates the Wasserstein distance between them for each layer and network.
"""

import torch
import numpy as np
import os
import glob
import yaml
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import concurrent.futures
import multiprocessing as mp
import gc

# Import ripser for persistence diagram computation
from ripser import ripser

# Import GUDHI for Wasserstein distance computation
try:
    from gudhi.wasserstein import wasserstein_distance
    GUDHI_AVAILABLE = True
except ImportError:
    print("WARNING: GUDHI not available. Trying alternative Wasserstein implementations...")
    GUDHI_AVAILABLE = False
    # Try to import from gudhi.hera as alternative
    try:
        from gudhi.hera import wasserstein_distance
    except ImportError:
        raise ImportError("Neither gudhi.wasserstein nor gudhi.hera available. Please install GUDHI.")

# Import distance computation and other utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.distance_computation import knn_geodesic_distance


@dataclass
class WassersteinResult:
    """Result structure for Wasserstein distance computation."""
    network_idx: int
    layer_idx: int
    wasserstein_distances: Dict[int, float]  # dimension -> distance
    train_persistence_diagram: Dict[int, np.ndarray]  # dimension -> diagram
    test_persistence_diagram: Dict[int, np.ndarray]   # dimension -> diagram
    train_betti_numbers: List[int]  # Betti numbers for train
    test_betti_numbers: List[int]   # Betti numbers for test
    computation_time: float
    success: bool
    error_message: Optional[str] = None


def load_config(config_path: str = "configs/homology_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_train_test_layer_outputs(input_dir: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Load train and test layer output files from the input directory.
    
    Returns:
        Tuple of (train_files_dict, test_files_dict)
    """
    train_files = {}
    test_files = {}
    
    # Look for train files
    train_pattern = os.path.join(input_dir, "*_train_layer_outputs.pt")
    for file_path in glob.glob(train_pattern):
        filename = os.path.basename(file_path)
        try:
            data = torch.load(file_path, map_location='cpu')
            if isinstance(data, dict) and 'layer_outputs' in data:
                train_files[filename] = data['layer_outputs']
            else:
                train_files[filename] = data
            print(f"  Loaded train file {filename}: {train_files[filename].shape}")
        except Exception as e:
            print(f"WARNING: Could not load {filename}: {e}")
    
    # Look for test files
    test_pattern = os.path.join(input_dir, "*_test_layer_outputs.pt")
    for file_path in glob.glob(test_pattern):
        filename = os.path.basename(file_path)
        try:
            data = torch.load(file_path, map_location='cpu')
            if isinstance(data, dict) and 'layer_outputs' in data:
                test_files[filename] = data['layer_outputs']
            else:
                test_files[filename] = data
            print(f"  Loaded test file {filename}: {test_files[filename].shape}")
        except Exception as e:
            print(f"WARNING: Could not load {filename}: {e}")
    
    return train_files, test_files


def compute_betti_numbers_from_diagram(diagram: np.ndarray, max_edge_length: float) -> int:
    """
    Compute Betti number from a persistence diagram.
    Count features that persist at the threshold value.
    """
    if len(diagram) == 0:
        return 0
    
    epsilon = 1e-10
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    
    # Count features that are born at or before max_edge_length and die after it (or are infinite)
    persistent_features = np.sum((births <= max_edge_length + epsilon) & 
                                ((deaths > max_edge_length + epsilon) | 
                                 (deaths == np.inf)))
    
    return int(persistent_features)


def compute_persistence_diagram(activations: np.ndarray, config: Dict) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """
    Compute persistence diagram for a set of activations using Ripser.
    
    Returns:
        Tuple of (persistence diagrams dict, betti numbers list)
    """
    # Apply normalization if enabled
    if config.get('computation', {}).get('normalize_data', True):
        activations = (activations - np.mean(activations, axis=0, keepdims=True)) / (np.std(activations, axis=0, keepdims=True) + 1e-8)
    
    # Apply sampling if dataset is too large
    sampling_config = config.get('sampling', {})
    max_points = sampling_config.get('fps_num_points', 1000)
    
    if len(activations) > max_points:
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(activations), max_points, replace=False)
        activations = activations[indices]
    
    # Compute distance matrix
    distance_matrix = knn_geodesic_distance(activations)
    
    # Compute persistent homology
    max_dimension = config.get('computation', {}).get('max_dimension', 1)
    max_edge_length = config.get('computation', {}).get('max_edge_length', 0.5)
    
    result = ripser(distance_matrix,
                   maxdim=max_dimension,
                   thresh=max_edge_length,
                   distance_matrix=True)
    
    # Extract persistence diagrams and compute Betti numbers
    diagrams = {}
    betti_numbers = []
    
    for dim in range(max_dimension + 1):
        if dim < len(result['dgms']):
            diagrams[dim] = result['dgms'][dim]
            betti = compute_betti_numbers_from_diagram(diagrams[dim], max_edge_length)
            betti_numbers.append(betti)
        else:
            diagrams[dim] = np.array([])
            betti_numbers.append(0)
    
    return diagrams, betti_numbers


def compute_wasserstein_distances(train_diagram: Dict[int, np.ndarray], 
                                 test_diagram: Dict[int, np.ndarray],
                                 order: float = 1.0,
                                 internal_p: float = np.inf) -> Dict[int, float]:
    """
    Compute Wasserstein distances between train and test persistence diagrams for each dimension.
    
    Parameters:
        train_diagram: Dictionary mapping dimension to persistence diagram
        test_diagram: Dictionary mapping dimension to persistence diagram
        order: Order of the Wasserstein distance (default 1.0)
        internal_p: Internal p-norm for ground metric (default infinity)
    
    Returns:
        Dictionary mapping dimension to Wasserstein distance
    """
    distances = {}
    
    for dim in train_diagram.keys():
        if dim not in test_diagram:
            distances[dim] = np.inf
            continue
        
        train_dgm = train_diagram[dim]
        test_dgm = test_diagram[dim]
        
        # Handle empty diagrams
        if len(train_dgm) == 0 and len(test_dgm) == 0:
            distances[dim] = 0.0
        elif len(train_dgm) == 0 or len(test_dgm) == 0:
            # One diagram is empty - distance is sum of all persistence values in non-empty diagram
            non_empty = train_dgm if len(train_dgm) > 0 else test_dgm
            # Filter out infinite death times
            finite_points = non_empty[non_empty[:, 1] != np.inf]
            if len(finite_points) > 0:
                persistences = finite_points[:, 1] - finite_points[:, 0]
                distances[dim] = np.sum(persistences ** order) ** (1.0 / order)
            else:
                distances[dim] = 0.0
        else:
            # Both diagrams non-empty - compute Wasserstein distance
            try:
                # Set keep_essential_parts=False to avoid warnings about infinite points
                dist = wasserstein_distance(train_dgm, test_dgm, 
                                          order=order, 
                                          internal_p=internal_p,
                                          keep_essential_parts=False)
                distances[dim] = dist
            except Exception as e:
                print(f"WARNING: Failed to compute Wasserstein distance for dimension {dim}: {e}")
                distances[dim] = np.nan
    
    return distances


def process_single_layer(train_activations: np.ndarray, 
                        test_activations: np.ndarray,
                        config: Dict,
                        network_idx: int,
                        layer_idx: int,
                        verbose: bool = False) -> WassersteinResult:
    """Process a single layer to compute Wasserstein distance between train and test."""
    start_time = time.time()
    
    try:
        # Compute persistence diagrams and Betti numbers
        train_diagram, train_betti = compute_persistence_diagram(train_activations, config)
        test_diagram, test_betti = compute_persistence_diagram(test_activations, config)
        
        # Print Betti numbers only in verbose mode
        if verbose:
            print(f"\n  Network {network_idx}, Layer {layer_idx}:")
            print(f"    Train Betti numbers: {train_betti}")
            print(f"    Test Betti numbers:  {test_betti}")
        
        # Compute Wasserstein distances
        wasserstein_order = config.get('wasserstein', {}).get('order', 1.0)
        internal_p = config.get('wasserstein', {}).get('internal_p', np.inf)
        
        distances = compute_wasserstein_distances(train_diagram, test_diagram, 
                                                 order=wasserstein_order,
                                                 internal_p=internal_p)
        
        if verbose:
            print(f"    Wasserstein distances: {[f'{d:.4f}' if not np.isnan(d) else 'NaN' for d in distances.values()]}")
        
        computation_time = time.time() - start_time
        
        return WassersteinResult(
            network_idx=network_idx,
            layer_idx=layer_idx,
            wasserstein_distances=distances,
            train_persistence_diagram=train_diagram,
            test_persistence_diagram=test_diagram,
            train_betti_numbers=train_betti,
            test_betti_numbers=test_betti,
            computation_time=computation_time,
            success=True
        )
        
    except Exception as e:
        return WassersteinResult(
            network_idx=network_idx,
            layer_idx=layer_idx,
            wasserstein_distances={},
            train_persistence_diagram={},
            test_persistence_diagram={},
            train_betti_numbers=[],
            test_betti_numbers=[],
            computation_time=time.time() - start_time,
            success=False,
            error_message=str(e)
        )


def compute_wasserstein_generalization(config_path: str = "configs/homology_config.yaml",
                                     input_dir: str = None) -> None:
    """
    Main function to compute Wasserstein distance between train and test persistence diagrams.
    """
    print("WASSERSTEIN GENERALIZATION ANALYSIS")
    print("=" * 50)
    print("Computing Wasserstein distances between train and test persistence diagrams...")
    print("=" * 50)
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Override input directory if specified
    if input_dir is None:
        input_dir = 'results/train_test_layer_outputs'
    
    # Create output directory
    output_dir = Path('results/wasserstein_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load train and test layer outputs
    print(f"\nLoading layer outputs from: {input_dir}")
    train_files, test_files = load_train_test_layer_outputs(input_dir)
    
    if not train_files or not test_files:
        print("ERROR: No train or test layer output files found.")
        return
    
    # Match train and test files
    matched_files = []
    for train_file in train_files:
        # Find corresponding test file
        test_file = train_file.replace('_train_', '_test_')
        if test_file in test_files:
            matched_files.append((train_file, test_file))
    
    if not matched_files:
        print("ERROR: No matching train/test file pairs found.")
        return
    
    print(f"\nFound {len(matched_files)} matching train/test file pairs.")
    
    # Process each file pair
    all_results = {}
    
    for train_file, test_file in matched_files:
        print(f"\nProcessing pair: {train_file} <-> {test_file}")
        
        train_outputs = train_files[train_file]
        test_outputs = test_files[test_file]
        
        # Convert to numpy if needed
        if isinstance(train_outputs, torch.Tensor):
            train_outputs = train_outputs.cpu().numpy()
        if isinstance(test_outputs, torch.Tensor):
            test_outputs = test_outputs.cpu().numpy()
        
        # Expected shape: [num_networks, num_layers, num_samples, layer_dim]
        if train_outputs.ndim != 4 or test_outputs.ndim != 4:
            print(f"WARNING: Unexpected shape. Train: {train_outputs.shape}, Test: {test_outputs.shape}")
            continue
        
        num_networks, num_layers, _, _ = train_outputs.shape
        print(f"Processing {num_networks} networks with {num_layers} layers each...")
        
        # Process all layers
        results = []
        total_tasks = num_networks * num_layers
        completed_tasks = 0
        
        # Use parallel processing if enabled
        use_parallel = config.get('parallel', {}).get('enabled', True)
        num_workers = min(mp.cpu_count(), config.get('parallel', {}).get('num_workers', mp.cpu_count()))
        
        if use_parallel and total_tasks > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_task = {}
                for net_idx in range(num_networks):
                    for layer_idx in range(num_layers):
                        train_layer = train_outputs[net_idx, layer_idx]
                        test_layer = test_outputs[net_idx, layer_idx]
                        
                        future = executor.submit(process_single_layer, 
                                               train_layer, test_layer, 
                                               config, net_idx, layer_idx)
                        future_to_task[future] = (net_idx, layer_idx)
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
                    completed_tasks += 1
                    
                    if completed_tasks % max(1, total_tasks // 10) == 0:
                        print(f"Progress: {completed_tasks}/{total_tasks} ({100*completed_tasks/total_tasks:.1f}%)")
        else:
            # Sequential processing
            for net_idx in range(num_networks):
                for layer_idx in range(num_layers):
                    train_layer = train_outputs[net_idx, layer_idx]
                    test_layer = test_outputs[net_idx, layer_idx]
                    
                    result = process_single_layer(train_layer, test_layer, 
                                                config, net_idx, layer_idx, verbose=True)
                    results.append(result)
                    completed_tasks += 1
                    
                    if completed_tasks % max(1, total_tasks // 10) == 0:
                        print(f"Progress: {completed_tasks}/{total_tasks} ({100*completed_tasks/total_tasks:.1f}%)")
        
        # Organize results
        file_prefix = train_file.replace('_train_layer_outputs.pt', '')
        all_results[file_prefix] = {
            'results': results,
            'num_networks': num_networks,
            'num_layers': num_layers,
            'train_shape': train_outputs.shape,
            'test_shape': test_outputs.shape
        }
    
    # Save results
    print("\nSaving results...")
    save_wasserstein_results(all_results, config, output_dir, time.time() - start_time)
    
    print(f"\nTotal computation time: {time.time() - start_time:.2f} seconds")
    print("=" * 50)
    print("WASSERSTEIN GENERALIZATION ANALYSIS COMPLETED")
    print("=" * 50)


def save_wasserstein_results(all_results: Dict, config: Dict, output_dir: Path, total_time: float):
    """Save Wasserstein distance results and generate summary statistics."""
    
    # Process results for each file
    for file_prefix, file_data in all_results.items():
        results = file_data['results']
        num_networks = file_data['num_networks']
        num_layers = file_data['num_layers']
        
        # Get maximum dimension
        max_dim = config.get('computation', {}).get('max_dimension', 1)
        
        # Initialize tensors for storing distances and Betti numbers
        wasserstein_tensor = np.zeros((num_networks, num_layers, max_dim + 1))
        train_betti_tensor = np.zeros((num_networks, num_layers, max_dim + 1), dtype=int)
        test_betti_tensor = np.zeros((num_networks, num_layers, max_dim + 1), dtype=int)
        
        # Fill in the tensors
        for result in results:
            if result.success:
                for dim, dist in result.wasserstein_distances.items():
                    wasserstein_tensor[result.network_idx, result.layer_idx, dim] = dist
                
                # Store Betti numbers
                for dim in range(len(result.train_betti_numbers)):
                    train_betti_tensor[result.network_idx, result.layer_idx, dim] = result.train_betti_numbers[dim]
                    test_betti_tensor[result.network_idx, result.layer_idx, dim] = result.test_betti_numbers[dim]
            else:
                # Use NaN for failed computations
                wasserstein_tensor[result.network_idx, result.layer_idx, :] = np.nan
        
        # Save results
        output_file = output_dir / f'{file_prefix}_wasserstein_distances.pt'
        torch.save({
            'wasserstein_distances': torch.tensor(wasserstein_tensor),
            'train_betti_numbers': torch.tensor(train_betti_tensor),
            'test_betti_numbers': torch.tensor(test_betti_tensor),
            'config': config,
            'num_networks': num_networks,
            'num_layers': num_layers,
            'file_prefix': file_prefix,
            'computation_time': total_time
        }, output_file)
        
        print(f"\nResults saved to: {output_file}")
        print(f"Wasserstein distances shape: {wasserstein_tensor.shape}")
        
        # Print summary statistics
        print(f"\nSummary for {file_prefix}:")
        print("-" * 40)
        for dim in range(max_dim + 1):
            dim_distances = wasserstein_tensor[:, :, dim]
            valid_distances = dim_distances[~np.isnan(dim_distances)]
            
            if len(valid_distances) > 0:
                print(f"  Dimension {dim}:")
                print(f"    Mean distance: {np.mean(valid_distances):.4f}")
                print(f"    Std deviation: {np.std(valid_distances):.4f}")
                print(f"    Min distance:  {np.min(valid_distances):.4f}")
                print(f"    Max distance:  {np.max(valid_distances):.4f}")
                
                # Layer-wise statistics
                print(f"    Layer-wise mean distances:")
                for layer_idx in range(num_layers):
                    layer_distances = dim_distances[:, layer_idx]
                    layer_valid = layer_distances[~np.isnan(layer_distances)]
                    if len(layer_valid) > 0:
                        print(f"      Layer {layer_idx}: {np.mean(layer_valid):.4f} (±{np.std(layer_valid):.4f})")
        
        # Print Betti numbers for each layer
        print(f"\n  Betti Numbers by Layer:")
        print("  " + "-" * 60)
        
        # Create header
        dim_labels = ["B0", "B1", "B2", "B3"][:max_dim + 1]
        header_parts = ["  Layer"]
        for label in dim_labels:
            header_parts.append(f"Train {label:>6}  Test {label:>6}")
        
        header = " | ".join(header_parts)
        print(header)
        print("  " + "-" * len(header))
        
        # Print layer-wise Betti numbers (averaged across networks)
        for layer_idx in range(num_layers):
            row_parts = [f"  {layer_idx:5d}"]
            
            for dim in range(max_dim + 1):
                # Average across networks for this layer
                train_betti_avg = np.mean(train_betti_tensor[:, layer_idx, dim])
                test_betti_avg = np.mean(test_betti_tensor[:, layer_idx, dim])
                
                # If all networks have the same value, show as integer
                train_betti_all = train_betti_tensor[:, layer_idx, dim]
                test_betti_all = test_betti_tensor[:, layer_idx, dim]
                
                if np.all(train_betti_all == train_betti_all[0]):
                    train_str = f"{int(train_betti_all[0]):6d}"
                else:
                    train_str = f"{train_betti_avg:6.1f}"
                    
                if np.all(test_betti_all == test_betti_all[0]):
                    test_str = f"{int(test_betti_all[0]):6d}"
                else:
                    test_str = f"{test_betti_avg:6.1f}"
                
                row_parts.append(f"{train_str}  {test_str}")
            
            row = " | ".join(row_parts)
            print(row)
        
        print("  " + "-" * len(header))
        
        # If there are multiple networks, also show the variation
        if num_networks > 1:
            print(f"\n  Note: Values shown are averages across {num_networks} networks.")
            print("  For detailed per-network results, see the saved tensor files.")
    
    # Save configuration used
    config_output = output_dir / 'wasserstein_config_used.yaml'
    with open(config_output, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    # Save computation log
    log_file = output_dir / 'wasserstein_computation.log'
    with open(log_file, 'w') as f:
        f.write("Wasserstein Generalization Analysis Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Computation time: {total_time:.2f} seconds\n")
        f.write(f"Configuration file: {config.get('config_file', 'Unknown')}\n")
        f.write(f"Input directory: {config.get('input_dir', 'Unknown')}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Files processed: {list(all_results.keys())}\n")
        f.write(f"GUDHI available: {GUDHI_AVAILABLE}\n")
        f.write(f"Wasserstein order: {config.get('wasserstein', {}).get('order', 1.0)}\n")
        f.write(f"Internal p-norm: {config.get('wasserstein', {}).get('internal_p', np.inf)}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute Wasserstein distance for network generalization analysis.")
    parser.add_argument("--config", type=str, default="configs/homology_config.yaml",
                       help="Path to the configuration file.")
    parser.add_argument("--input-dir", type=str, default="results/train_test_layer_outputs",
                       help="Input directory containing train/test layer outputs.")
    
    args = parser.parse_args()
    compute_wasserstein_generalization(args.config, args.input_dir)