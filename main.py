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
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configurations
    training_config = load_config('configs/training_config.yaml')
    homology_config = load_config('configs/homology_config.yaml')
    viz_config = load_config('configs/visualization_config.yaml')
    
    # Create results directories if they don't exist
    for dir_path in ['results/models', 'results/plots', 'results/homology']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate dataset
    print("Generating dataset...")
    X, y = generate(n=1000, big_radius=3, small_radius=1)
    plot_torus_points(X, y)
    
    # Step 2: Train neural network
    print("\nTraining neural network...")
    device = torch.device(training_config['training']['device'])
    layer_outputs = train(
        X, y,
        width=training_config['model']['width'],
        layers=training_config['model']['layers'],
        epochs=training_config['training']['epochs'],
        batch_size=training_config['training']['batch_size'],
        lr=training_config['training']['learning_rate'],
        device=device
    )
    
    # Step 3: Compute homology for each layer
    print("\nComputing persistent homology...")
    betti_numbers = []
    for i, layer_output in enumerate(layer_outputs):
        print(f"Processing layer {i+1}/{len(layer_outputs)}")
        # Compute distance matrix for the layer
        from src.utils.distance import compute_distance_matrix
        dist_matrix = compute_distance_matrix(layer_output)
        
        # Compute persistent homology
        betti = compute_persistent_homology(
            dist_matrix,
            max_dimension=homology_config['computation']['max_dimension'],
            max_edge_length=homology_config['computation']['max_edge_length']
        )
        betti_numbers.append(betti)
    
    # Save results
    np.save('results/homology/betti_numbers.npy', np.array(betti_numbers))
    
    # Step 4: Visualize results
    print("\nGenerating visualizations...")
    plot_curves_from_h5(['results/homology/betti_numbers.npy'])
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 