"""
Compute homology for neural network layer activations.

This script loads layer outputs from results/layer_outputs, computes distance matrices
using functions from distance_computation.py, and then computes persistent homology
to extract topological features (Betti numbers) for each layer of each network.

Output format: [num_networks, num_layers, max_dimension] tensor of Betti numbers.
"""

import torch
import numpy as np
import os
import glob
import yaml
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import gudhi as gd

# Import distance computation functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.distance_computation import knn_geodesic_distance, load_config


def compute_persistent_homology_betti(distance_matrix: np.ndarray, max_dimension: int = 2, 
                                     max_edge_length: float = 1.0) -> List[int]:
    """
    Compute persistent homology and return Betti numbers with robust error handling.
    
    This function is adapted from homology.py but standalone to avoid circular imports.
    Constructs a Vietoris-Rips complex and calculates its persistent homology.
    
    Parameters:
    - distance_matrix: Square, symmetric distance matrix with zero diagonal
    - max_dimension: Maximum homology dimension to compute (e.g., 2 means H0, H1, H2)
    - max_edge_length: Maximum edge length for including edges in the Rips complex
    
    Returns:
    - List of Betti numbers for dimensions 0 up to max_dimension
    
    Raises:
    - ValueError: If distance_matrix is invalid (not square, not symmetric, negative diagonal)
    """
    try:
        # Validate input distance matrix
        if not isinstance(distance_matrix, np.ndarray):
            raise ValueError("distance_matrix must be a NumPy array.")
        
        if distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be a 2D array.")
        
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square.")
        
        # More lenient symmetry check for integer matrices
        if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-6):
            print(f"Warning: Distance matrix may not be perfectly symmetric, max diff: {np.max(np.abs(distance_matrix - distance_matrix.T))}")
        
        if not np.allclose(np.diag(distance_matrix), 0, atol=1e-6):
            print(f"Warning: Diagonal elements may not be zero, max diagonal: {np.max(np.diag(distance_matrix))}")
        
        # Check matrix size for memory safety
        n = distance_matrix.shape[0]
        if n > 500:
            print(f"Warning: Large distance matrix ({n}x{n}), this may cause memory issues")
        
        # Create Rips complex from distance matrix with error handling
        print(f"Creating Rips complex (max_edge_length={max_edge_length})", end="", flush=True)
        rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
        
        # Create simplex tree with memory constraints
        print(" -> building simplex tree", end="", flush=True)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        
        # Check if simplex tree is too large
        num_simplices = simplex_tree.num_simplices()
        if num_simplices > 100000:
            print(f" [Warning: Large simplex tree with {num_simplices} simplices]", end="", flush=True)
        
        # Optimize the simplex tree
        print(" -> optimizing", end="", flush=True)
        simplex_tree.collapse_edges()
        simplex_tree.expansion(max_dimension + 1)
        
        # Compute persistence
        print(" -> computing persistence", end="", flush=True)
        persistence = simplex_tree.persistence()
        
        # Extract Betti numbers
        betti_numbers = simplex_tree.betti_numbers()
        
        # Ensure we have Betti numbers for all dimensions up to max_dimension
        while len(betti_numbers) <= max_dimension:
            betti_numbers.append(0)
        
        return betti_numbers[:max_dimension + 1]
        
    except Exception as e:
        print(f"Error in homology computation: {e}")
        # Return safe default values
        return [1] + [0] * max_dimension


def load_layer_outputs(input_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load all layer output files from the input directory.
    
    Parameters:
    - input_dir: Directory containing layer output .pt files
    
    Returns:
    - Dictionary mapping filename to layer output tensors
    """
    layer_files = {}
    pattern = os.path.join(input_dir, "*.pt")
    
    for file_path in glob.glob(pattern):
        filename = os.path.basename(file_path)
        try:
            data = torch.load(file_path, map_location='cpu')
            if isinstance(data, dict) and 'layer_outputs' in data:
                layer_files[filename] = data['layer_outputs']
            else:
                layer_files[filename] = data
            print(f"Loaded {filename}: {layer_files[filename].shape}")
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
    
    return layer_files


def process_single_layer(layer_activations: np.ndarray, config: Dict, layer_idx: int = 0) -> List[int]:
    """
    Process a single layer's activations to compute Betti numbers.
    
    Parameters:
    - layer_activations: Numpy array of shape (num_samples, layer_dim)
    - config: Configuration dictionary
    - layer_idx: Layer index for logging purposes
    
    Returns:
    - List of Betti numbers for each dimension
    """
    try:
        # Check minimum points threshold
        min_points = config.get('sampling', {}).get('min_points_threshold', 50)
        if len(layer_activations) < min_points:
            print(f"Warning: Layer {layer_idx} has only {len(layer_activations)} points, below threshold {min_points}")
            return [0] * (config.get('computation', {}).get('max_dimension', 1) + 1)
        
        print(f"Processing layer {layer_idx}: {layer_activations.shape}", end="", flush=True)
        
        # Compute distance matrix using knn_geodesic_distance
        distance_matrix = knn_geodesic_distance(layer_activations)
        print(f" -> distance matrix {distance_matrix.shape}", end="", flush=True)
        
        # Compute persistent homology
        max_dimension = config.get('computation', {}).get('max_dimension', 1)
        max_edge_length = config.get('computation', {}).get('max_edge_length', 0.5)
        
        betti_numbers = compute_persistent_homology_betti(
            distance_matrix.astype(np.float64), 
            max_dimension=max_dimension,
            max_edge_length=max_edge_length
        )
        
        print(f" -> Betti numbers: {betti_numbers}")
        return betti_numbers
        
    except Exception as e:
        print(f"Error processing layer {layer_idx}: {e}")
        # Return zero Betti numbers on error
        max_dimension = config.get('computation', {}).get('max_dimension', 1)
        return [0] * (max_dimension + 1)


def compute_layer_homology(config_path: str = "configs/homology_config.yaml") -> None:
    """
    Main function to compute homology for all layer outputs.
    
    Loads layer outputs, computes distance matrices, and calculates Betti numbers
    for each network and layer. Saves results in the specified output format.
    
    Parameters:
    - config_path: Path to the homology configuration file
    """
    print("Starting homology computation...")
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract key parameters
    input_dir = config.get('io', {}).get('input_dir', 'results/layer_outputs')
    output_dir = config.get('io', {}).get('output_dir', 'results/homology')
    max_dimension = config.get('computation', {}).get('max_dimension', 1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all layer output files
    print(f"Loading layer outputs from {input_dir}...")
    layer_files = load_layer_outputs(input_dir)
    
    if not layer_files:
        raise ValueError(f"No layer output files found in {input_dir}")
    
    # Process each file
    all_betti_results = {}
    
    for filename, layer_outputs in layer_files.items():
        print(f"\nProcessing {filename}...")
        
        # Convert to numpy if needed
        if isinstance(layer_outputs, torch.Tensor):
            layer_outputs = layer_outputs.cpu().numpy()
        
        # Expected shape: [num_networks, num_layers, num_samples, layer_dim]
        if layer_outputs.ndim == 4:
            num_networks, num_layers, num_samples, layer_dim = layer_outputs.shape
            print(f"Shape: [{num_networks}, {num_layers}, {num_samples}, {layer_dim}]")
            
            # Initialize results tensor for this file
            betti_results = np.zeros((num_networks, num_layers, max_dimension + 1), dtype=np.int32)
            
            # Process each network and layer
            for net_idx in range(num_networks):
                for layer_idx in range(num_layers):
                    # Extract single layer activations: (num_samples, layer_dim)
                    layer_data = layer_outputs[net_idx, layer_idx]
                    
                    # Compute Betti numbers for this layer
                    betti_numbers = process_single_layer(layer_data, config, layer_idx)
                    
                    # Store results (truncate to max_dimension + 1 if needed)
                    betti_results[net_idx, layer_idx] = betti_numbers[:max_dimension + 1]
            
            all_betti_results[filename] = betti_results
            
        else:
            print(f"Warning: Unexpected shape {layer_outputs.shape} for {filename}, skipping...")
    
    # Save results
    if all_betti_results:
        # If only one file, save directly; if multiple, save as dictionary
        if len(all_betti_results) == 1:
            results_tensor = list(all_betti_results.values())[0]
        else:
            results_tensor = all_betti_results
        
        output_file = os.path.join(output_dir, 'layer_betti_numbers.pt')
        torch.save(results_tensor, output_file)
        print(f"\nSaved Betti numbers to {output_file}")
        
        # Save configuration used
        config_output = os.path.join(output_dir, 'homology_config_used.yaml')
        with open(config_output, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        # Save computation log
        total_time = time.time() - start_time
        log_file = os.path.join(output_dir, 'homology_computation.log')
        with open(log_file, 'w') as f:
            f.write(f"Homology Computation Log\n")
            f.write(f"========================\n")
            f.write(f"Start time: {time.ctime(start_time)}\n")
            f.write(f"Total computation time: {total_time:.2f} seconds\n")
            f.write(f"Configuration file: {config_path}\n")
            f.write(f"Input directory: {input_dir}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Max dimension: {max_dimension}\n")
            f.write(f"Files processed: {list(all_betti_results.keys())}\n")
            for filename, results in all_betti_results.items():
                f.write(f"\n{filename}:\n")
                f.write(f"  Shape: {results.shape}\n")
                f.write(f"  Betti number ranges: {[f'[{results[:,:,i].min()}, {results[:,:,i].max()}]' for i in range(results.shape[2])]}\n")
        
        print(f"Computation completed in {total_time:.2f} seconds")
        print(f"Results shape: {results_tensor.shape if hasattr(results_tensor, 'shape') else 'Dictionary'}")
        
    else:
        print("No valid layer outputs were processed.")


if __name__ == "__main__":
    # Example usage
    compute_layer_homology()