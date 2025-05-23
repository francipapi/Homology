import os
import sys
import time
import yaml
import numpy as np
import mlx.core as mx

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.data import dataset
from src.models.trainer_mlx_v2 import train as train_original
from src.models.vectorised_mlx import train_vectorized

def test_with_dataset(dataset_name: str, X: np.ndarray, y: np.ndarray, config_dict: dict):
    """Test both implementations with a given dataset."""
    print(f"\nTesting with {dataset_name} dataset...")
    
    # Test original implementation
    print("\nOriginal Implementation:")
    start_time = time.time()
    original_partials = train_original([X, y])
    original_time = time.time() - start_time
    print(f"Training time: {original_time:.2f} seconds")
    
    # Test vectorized implementation
    print("\nVectorized Implementation:")
    start_time = time.time()
    vectorized_model, accuracies = train_vectorized(X, y, config_dict)
    vectorized_time = time.time() - start_time
    print(f"Training time: {vectorized_time:.2f} seconds")
    
    # Get partial outputs from vectorized model
    print("\nTesting partial outputs operations:")
    X_mlx = mx.array(X)
    partials = vectorized_model.get_full_partials(X_mlx)
    
    # Print shape information
    print(f"\nPartial outputs shape: {partials.shape}")
    print(f"Number of models: {partials.shape[0]}")
    print(f"Number of layers: {partials.shape[1]}")
    print(f"Dataset size: {partials.shape[2]}")
    print(f"Hidden dimension: {partials.shape[3]}")
    
    # Demonstrate tensor operations
    print("\nDemonstrating tensor operations:")
    
    # 1. Get outputs for a specific model and layer
    model_idx = 0
    layer_idx = 1
    specific_outputs = partials[model_idx, layer_idx]
    print(f"\n1. Outputs for model {model_idx}, layer {layer_idx}:")
    print(f"Shape: {specific_outputs.shape}")
    print(f"Mean activation: {mx.mean(specific_outputs).item():.4f}")
    
    # 2. Get outputs for all models at a specific layer
    layer_outputs = partials[:, layer_idx]
    print(f"\n2. Outputs for all models at layer {layer_idx}:")
    print(f"Shape: {layer_outputs.shape}")
    print(f"Mean activation per model: {np.array(mx.mean(layer_outputs, axis=(1, 2)))}")
    
    # 3. Get outputs for a specific model across all layers
    model_outputs = partials[model_idx]
    print(f"\n3. Outputs for model {model_idx} across all layers:")
    print(f"Shape: {model_outputs.shape}")
    print(f"Mean activation per layer: {np.array(mx.mean(model_outputs, axis=(1)))}")
    
    # 4. Compare with original implementation
    print("\n4. Comparing with original implementation:")
    if original_partials is not None:
        print("Original partials shapes:", [p.shape for p in original_partials])
        original_shapes = [p.shape for p in original_partials]
    else:
        print("Original implementation did not return partials (likely due to low accuracy).")
        original_shapes = None
    print("Vectorized partials shape:", partials.shape)
    
    # Compare performance
    speedup = original_time / vectorized_time
    print(f"\nPerformance Comparison:")
    print(f"Speedup factor: {speedup:.2f}x")
    
    return {
        'original_time': original_time,
        'vectorized_time': vectorized_time,
        'speedup': speedup,
        'original_shapes': original_shapes,
        'vectorized_shape': partials.shape,
        'model_activations': np.array(mx.mean(partials, axis=(1, 2, 3))),  # Mean activation per model
        'layer_activations': np.array(mx.mean(partials, axis=(0, 2, 3)))   # Mean activation per layer
    }

def main():
    # Load configuration
    with open('configs/training_config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create results directory
    os.makedirs('test_results', exist_ok=True)
    
    # Generate easy dataset
    X_easy, y_easy = dataset.generate(
        n=4000,
        big_radius=3,
        small_radius=1
    )
    
    # Generate complex dataset
    X_complex, y_complex = dataset.generate(
        n=4000,
        big_radius=5,
        small_radius=2
    )
    
    # Run tests
    easy_results = test_with_dataset("Easy", X_easy, y_easy, config_dict)
    complex_results = test_with_dataset("Complex", X_complex, y_complex, config_dict)
    
    # Save results
    with open('test_results/vectorized_comparison_results.txt', 'w') as f:
        f.write("Vectorized vs Original Implementation Comparison\n")
        f.write("==============================================\n\n")
        
        f.write("Easy Dataset Results:\n")
        f.write(f"Original Time: {easy_results['original_time']:.2f} seconds\n")
        f.write(f"Vectorized Time: {easy_results['vectorized_time']:.2f} seconds\n")
        f.write(f"Speedup: {easy_results['speedup']:.2f}x\n")
        f.write("\nShape Comparison:\n")
        f.write("Original: " + str(easy_results['original_shapes']) + "\n")
        f.write("Vectorized: " + str(easy_results['vectorized_shape']) + "\n")
        f.write("\nModel Activations (Easy):\n")
        f.write(str(easy_results['model_activations']) + "\n")
        f.write("\nLayer Activations (Easy):\n")
        f.write(str(easy_results['layer_activations']) + "\n\n")
        
        f.write("Complex Dataset Results:\n")
        f.write(f"Original Time: {complex_results['original_time']:.2f} seconds\n")
        f.write(f"Vectorized Time: {complex_results['vectorized_time']:.2f} seconds\n")
        f.write(f"Speedup: {complex_results['speedup']:.2f}x\n")
        f.write("\nShape Comparison:\n")
        f.write("Original: " + str(complex_results['original_shapes']) + "\n")
        f.write("Vectorized: " + str(complex_results['vectorized_shape']) + "\n")
        f.write("\nModel Activations (Complex):\n")
        f.write(str(complex_results['model_activations']) + "\n")
        f.write("\nLayer Activations (Complex):\n")
        f.write(str(complex_results['layer_activations']) + "\n")

if __name__ == "__main__":
    main() 