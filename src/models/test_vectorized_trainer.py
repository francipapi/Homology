import os
import sys
import time
import yaml
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.datasets import make_blobs

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.data import dataset
from src.models.vectorised_mlx import train_vectorized, VectorizedMLP

def analyze_model_performance(
    model: VectorizedMLP,
    X: mx.array,
    y: mx.array,
    batch_size: int = 1024
) -> Dict:
    """Analyze model performance with detailed metrics."""
    # Get predictions and partial outputs
    logits, partials = model.full_forward_pass(X, batch_size)
    predictions = mx.argmax(logits, axis=-1)
    
    # Ensure y is 1D
    if len(y.shape) > 1:
        y = y.reshape(-1)
    
    # Compute accuracy per model
    y_broadcast = mx.broadcast_to(y, (model.num_models, y.shape[0]))
    accuracies = mx.mean(predictions == y_broadcast, axis=1)
    
    # Compute mean activations
    mean_activations = mx.mean(partials, axis=(2, 3))  # Shape: (num_models, num_layers)
    
    # Compute activation statistics
    activation_stats = {
        'mean': mx.mean(partials, axis=(0, 2, 3)),  # Per layer
        'std': mx.std(partials, axis=(0, 2, 3)),    # Per layer
        'max': mx.max(partials, axis=(0, 2, 3)),    # Per layer
        'min': mx.min(partials, axis=(0, 2, 3))     # Per layer
    }
    
    return {
        'accuracies': np.array(accuracies),
        'mean_activations': np.array(mean_activations),
        'activation_stats': {k: np.array(v) for k, v in activation_stats.items()},
        'predictions': np.array(predictions),
        'partials': np.array(partials)
    }

def plot_training_curves(accuracies: List[np.ndarray], save_path: str):
    """Plot training curves for all models."""
    plt.figure(figsize=(12, 6))
    
    # Plot individual model accuracies
    for i, model_acc in enumerate(accuracies):
        plt.plot(model_acc, label=f'Model {i}', alpha=0.3)
    
    # Plot mean accuracy
    mean_acc = np.mean(accuracies, axis=0)
    plt.plot(mean_acc, label='Mean Accuracy', linewidth=2, color='black')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Curves for All Models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_activation_heatmap(mean_activations: np.ndarray, save_path: str):
    """Plot heatmap of mean activations across models and layers."""
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_activations, aspect='auto', cmap='viridis')
    plt.colorbar(label='Mean Activation')
    plt.xlabel('Layer')
    plt.ylabel('Model')
    plt.title('Mean Activations Across Models and Layers')
    plt.savefig(save_path)
    plt.close()

def plot_dataset(X, y, title, save_path=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.5)
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.title(title)
    plt.colorbar(scatter, label='Label')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def test_vectorized_trainer(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    config_dict: dict,
    results_dir: str
) -> Dict:
    """Test vectorized trainer with comprehensive analysis."""
    print(f"\nTesting vectorized trainer on {dataset_name} dataset...")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Train model
    start_time = time.time()
    model, accuracies = train_vectorized(X, y, config_dict)
    training_time = time.time() - start_time
    
    # Convert data to MLX arrays for analysis
    X_mlx = mx.array(X)
    y_mlx = mx.array(y.astype(np.int32))
    
    # Analyze performance
    performance = analyze_model_performance(model, X_mlx, y_mlx)
    
    # Print detailed partial outputs analysis
    print("\nDetailed Partial Outputs Analysis:")
    print("=================================")
    partials = performance['partials']
    print(f"Full partials shape: {partials.shape}")
    print(f"Number of models: {partials.shape[0]}")
    print(f"Number of layers: {partials.shape[1]}")
    print(f"Dataset size: {partials.shape[2]}")
    print(f"Hidden dimension: {partials.shape[3]}")
    
    # Print statistics for each layer
    print("\nLayer-wise Statistics:")
    print("====================")
    for layer_idx in range(partials.shape[1]):
        layer_outputs = partials[:, layer_idx]  # Shape: (num_models, dataset_size, hidden_dim)
        print(f"\nLayer {layer_idx}:")
        print(f"  Shape: {layer_outputs.shape}")
        print(f"  Mean activation: {np.mean(layer_outputs):.4f}")
        print(f"  Std activation: {np.std(layer_outputs):.4f}")
        print(f"  Max activation: {np.max(layer_outputs):.4f}")
        print(f"  Min activation: {np.min(layer_outputs):.4f}")
    
    # Run full forward pass on entire dataset
    print("\nRunning full forward pass on entire dataset...")
    final_outputs, full_partials = model.full_forward_pass(X_mlx)
    print(f"Final outputs shape: {final_outputs.shape}")
    print(f"Full partials shape: {full_partials.shape}")
    
    # Plot training curves
    plot_training_curves(
        accuracies,
        os.path.join(results_dir, f'{dataset_name}_training_curves.png')
    )
    
    # Plot activation heatmap
    plot_activation_heatmap(
        performance['mean_activations'],
        os.path.join(results_dir, f'{dataset_name}_activation_heatmap.png')
    )
    
    # Save detailed results
    results = {
        'training_time': training_time,
        'final_accuracies': performance['accuracies'],
        'mean_accuracy': float(np.mean(performance['accuracies'])),
        'std_accuracy': float(np.std(performance['accuracies'])),
        'best_accuracy': float(np.max(performance['accuracies'])),
        'worst_accuracy': float(np.min(performance['accuracies'])),
        'activation_stats': performance['activation_stats'],
        'partials_shape': partials.shape,
        'full_forward_shape': {
            'final_outputs': final_outputs.shape,
            'full_partials': full_partials.shape
        },
        'layer_stats': {
            f'layer_{i}': {
                'mean': float(np.mean(partials[:, i])),
                'std': float(np.std(partials[:, i])),
                'max': float(np.max(partials[:, i])),
                'min': float(np.min(partials[:, i]))
            }
            for i in range(partials.shape[1])
        }
    }
    
    # Print summary
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Mean accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Best model accuracy: {results['best_accuracy']:.4f}")
    print(f"Worst model accuracy: {results['worst_accuracy']:.4f}")
    
    return results

def main():
    # Load configuration
    with open('configs/training_config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create results directory
    results_dir = 'test_results/vectorized_trainer'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate datasets
    datasets = {
        'Easy': dataset.generate(n=4000, big_radius=3, small_radius=1),
        'Complex': dataset.generate(n=4000, big_radius=5, small_radius=2)
    }
    # Add a simple synthetic dataset
    X_blob, y_blob = make_blobs(n_samples=2000, centers=2, n_features=3, cluster_std=2.0, random_state=42)
    datasets['Blobs'] = (X_blob, y_blob)
    
    # Test on each dataset
    results = {}
    for name, (X, y) in datasets.items():
        plot_dataset(X, y, f'{name} Dataset', os.path.join(results_dir, f'{name}_dataset.png'))
        results[name] = test_vectorized_trainer(name, X, y, config_dict, results_dir)
    
    # Save summary results
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write("Vectorized Trainer Test Results\n")
        f.write("=============================\n\n")
        
        for name, result in results.items():
            f.write(f"{name} Dataset:\n")
            f.write(f"Training Time: {result['training_time']:.2f} seconds\n")
            f.write(f"Mean Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}\n")
            f.write(f"Best Accuracy: {result['best_accuracy']:.4f}\n")
            f.write(f"Worst Accuracy: {result['worst_accuracy']:.4f}\n")
            f.write("\nActivation Statistics:\n")
            for stat_name, values in result['activation_stats'].items():
                f.write(f"{stat_name}: {values}\n")
            f.write("\n" + "="*50 + "\n\n")

if __name__ == "__main__":
    main() 