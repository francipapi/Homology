"""
Main script to run the topological data analysis pipeline.

This script orchestrates the entire process:
1. Loads configuration settings from YAML files.
2. Creates necessary directories for storing results.
3. Generates a dataset (e.g., points on a torus).
4. Trains a neural network model on the generated dataset.
5. Extracts activations from the trained model's layers.
6. Computes persistent homology for the activations of each layer.
7. Saves the Betti numbers and other homology outputs (diagrams, plots).
8. Visualizes the Betti curves.
"""
import os
import yaml
import torch
import numpy as np
from pathlib import Path

# Import from restructured modules
from src.data.dataset import generate, plot_torus_points
from src.models.trainer import train
from src.topology.homology import compute_persistent_homology
from src.visualization.plot_curves import plot_curves_from_h5

def load_config(config_path):
    """
    Load configuration settings from a YAML file.

    Args:
        config_path (str or Path): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """
    Main function to execute the TDA pipeline.
    """
    # --- Configuration Loading ---
    print("Loading configurations...")
    # Load parameters for training, homology computation, and visualization.
    training_config = load_config('configs/training_config.yaml')
    homology_config = load_config('configs/homology_config.yaml')
    # viz_config is loaded but not explicitly used in this main script's current flow.
    # It might be intended for other visualization scripts or future enhancements.
    viz_config = load_config('configs/visualization_config.yaml')
    
    # --- Directory Setup ---
    # Define and create directories for storing results if they don't already exist.
    # This helps in organizing outputs like trained models, plots, and homology data.
    results_dirs = ['results/models', 'results/plots', 'results/homology']
    for dir_path in results_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # --- Step 1: Dataset Generation ---
    print("Step 1: Generating dataset...")
    # Retrieve data generation parameters from the training configuration.
    data_gen_params = training_config['data']['generation']
    X, y = generate(
        n=data_gen_params['n'],
        big_radius=data_gen_params['big_radius'],
        small_radius=data_gen_params['small_radius']
    )
    # Plot the generated dataset (e.g., torus points) for visual inspection.
    # The plot is saved to 'results/plots/torus_dataset.png'.
    plot_torus_points(X, y, filename=Path(results_dirs[1]) / "torus_dataset.png")
    print(f"Dataset generated with {X.shape[0]} points. Plot saved to {results_dirs[1]}/torus_dataset.png")
    
    # --- Step 2: Neural Network Training ---
    print("\nStep 2: Training neural network...")
    # The `train` function handles device selection based on `training_config['training']['device']`.
    # It also uses the `seed` from `training_config['training']['seed']` for reproducibility.
    layer_outputs = train(
        X, y,
        width=training_config['model']['width'],
        layers=training_config['model']['layers'],
        epochs=training_config['training']['epochs'],
        batch_size=training_config['training']['batch_size'],
        lr=training_config['training']['learning_rate'],
        device=training_config['training']['device'], # Pass the device string ('auto', 'cpu', 'cuda', 'mps')
        seed=training_config['training']['seed']      # Pass the random seed for training
    )
    # layer_outputs contains the activations from the ReLU layers of the trained model.
    print("Neural network training completed. Layer outputs extracted.")
    
    # --- Step 3: Persistent Homology Computation ---
    print("\nStep 3: Computing persistent homology...")
    all_betti_numbers = []  # List to store Betti numbers for each layer.
    
    # Define base path for saving homology-related results.
    homology_results_path = Path(results_dirs[2]) # results_dirs[2] is 'results/homology'

    # Check if layer_outputs exist and are not empty.
    if not hasattr(layer_outputs, 'size') or layer_outputs.size == 0:
        print("Warning: No layer outputs found or layer_outputs is empty. Skipping Homology Computation (Step 3) and Visualization (Step 4).")
    else:
        # Iterate through the activations of each layer.
        for i, layer_activation_data in enumerate(layer_outputs):
            print(f"  Processing Layer {i+1}/{len(layer_outputs)} for homology...")
            
            # Compute the distance matrix for the current layer's activation data.
            # The distance matrix is a prerequisite for constructing the Rips complex.
            from src.utils.graph import distance
            dist_matrix = distance(layer_activation_data, k=homology_config['computation']['num_neighbors'])
            print(f"    Distance matrix computed for layer {i+1}, shape: {dist_matrix.shape}.")
            
            # Define specific filenames for this layer's homology outputs.
            diag_filepath = homology_results_path / f"layer_{i+1}_persistence_diagram.txt"
            plot_filepath = homology_results_path / f"layer_{i+1}_barcode_plot.png"
            betti_filepath = homology_results_path / f"layer_{i+1}_betti_numbers.txt"
            
            # Compute persistent homology using parameters from homology_config.
            # This function will also save the diagram, plot, and Betti numbers if filenames are provided.
            betti_for_layer = compute_persistent_homology(
                dist_matrix,
                max_dimension=homology_config['computation']['max_dimension'],
                max_edge_length=homology_config['computation']['max_edge_length'],
                diag_filename=str(diag_filepath),        # Pass path as string
                plot_barcode_filename=str(plot_filepath),# Pass path as string
                betti_filename=str(betti_filepath)       # Pass path as string
            )
            all_betti_numbers.append(betti_for_layer)
            print(f"    Homology computed for layer {i+1}. Betti numbers: {betti_for_layer}")
        
        # Save all collected Betti numbers into a single .npy file for later visualization or analysis.
        # This file will contain a list of lists (or a 2D array if Betti numbers per layer are consistent in length).
        betti_summary_filepath = homology_results_path / "all_layers_betti_numbers.npy"
        # Using dtype=object for np.array to handle potentially ragged arrays (if Betti vectors have different lengths)
        np.save(betti_summary_filepath, np.array(all_betti_numbers, dtype=object))
        print(f"All Betti numbers for all layers saved to {betti_summary_filepath}")
    
        # --- Step 4: Results Visualization ---
        print("\nStep 4: Generating Betti curve visualizations...")
        # Plot Betti curves from the saved .npy file containing Betti numbers for all layers.
        # The plot_curves_from_h5 function is versatile; here it's used for .npy files.
        # (The 'h5' in its name might be historical or indicate broader capabilities).
        betti_curves_plot_path = Path(results_dirs[1]) / "betti_curves.png" # results_dirs[1] is 'results/plots'
        plot_curves_from_h5(
            [str(betti_summary_filepath)], # The function expects a list of file paths.
            output_path=str(betti_curves_plot_path),
            # Additional visualization parameters could be passed here if needed,
            # potentially sourced from viz_config.
        )
        print(f"Betti curves visualization generated and saved to {betti_curves_plot_path}")
    
    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    # This standard Python construct ensures that main() is called only when the script
    # is executed directly (e.g., `python main.py`), not when imported as a module.
    main() 