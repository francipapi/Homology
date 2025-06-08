import os
import sys
import time
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.data import dataset
from src.models.trainer_mlx_parallel import train_models_in_batches, TrainingConfig
import yaml

def test_with_dataset(dataset_name, X, y, config_dict):
    """Test the trainer with a specific dataset."""
    print(f"\n{'='*50}")
    print(f"Testing with {dataset_name} dataset")
    print(f"{'='*50}")
    
    # Create training configuration
    config = TrainingConfig(
        num_layers=config_dict['model']['layers'],
        hidden_dim=config_dict['model']['width'],
        num_classes=2,
        batch_size=config_dict['training']['batch_size'],
        num_epochs=config_dict['training']['epochs'],
        learning_rate=config_dict['training']['learning_rate'],
        decay_coeff=config_dict['training']['decay_coeff'],
        activation=config_dict['model']['activation'],
        device=config_dict['training']['device'],
        verbose=True
    )
    
    # Train models
    start = time.time()
    results = train_models_in_batches(X, y, config_dict, config)
    end = time.time()
    
    # Calculate statistics
    accuracies = [accuracy for _, accuracy, _ in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    best_acc = max(accuracies)
    worst_acc = min(accuracies)
    
    # Print results
    print(f"\nResults for {dataset_name}:")
    print(f"Training time: {end - start:.2f} seconds")
    print(f"Average accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Best accuracy: {best_acc:.3f}")
    print(f"Worst accuracy: {worst_acc:.3f}")
    
    return {
        'dataset': dataset_name,
        'time': end - start,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'best_acc': best_acc,
        'worst_acc': worst_acc
    }

def main():
    # Load configuration
    config_path = 'configs/training_config.yaml'
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create results directory
    os.makedirs('test_results', exist_ok=True)
    
    # Test with easy dataset
    print("\nGenerating easy dataset...")
    X_easy, y_easy = dataset.gen_easy(
        n=config_dict['data']['generation']['n'],
        big_radius=config_dict['data']['generation']['big_radius'],
        small_radious=config_dict['data']['generation']['small_radius']
    )
    
    # Test with complex dataset
    print("\nGenerating complex dataset...")
    X_complex, y_complex = dataset.generate(
        n=config_dict['data']['generation']['n'],
        big_radius=config_dict['data']['generation']['big_radius'],
        small_radius=config_dict['data']['generation']['small_radius']
    )
    
    # Run tests
    results = []
    results.append(test_with_dataset("Easy", X_easy, y_easy, config_dict))
    results.append(test_with_dataset("Complex", X_complex, y_complex, config_dict))
    
    # Save test results
    results_path = 'test_results/trainer_mlx_parallel_test_results.txt'
    with open(results_path, 'w') as f:
        f.write("Trainer MLX Parallel Test Results\n")
        f.write("================================\n\n")
        
        for result in results:
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"Training time: {result['time']:.2f} seconds\n")
            f.write(f"Average accuracy: {result['mean_acc']:.3f} ± {result['std_acc']:.3f}\n")
            f.write(f"Best accuracy: {result['best_acc']:.3f}\n")
            f.write(f"Worst accuracy: {result['worst_acc']:.3f}\n")
            f.write("\n")
    
    print(f"\nTest results saved to {results_path}")

if __name__ == "__main__":
    main() 