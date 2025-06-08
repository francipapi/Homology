import argparse
import time
from functools import partial
from typing import List, Tuple, Dict, Any
import concurrent.futures
import multiprocessing
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data import dataset
import yaml
import os

class MLP(nn.Module):
    """A simple MLP with configurable architecture."""

    def __init__(
        self, 
        num_layers: int, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        activation: str = "relu"
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.activation = getattr(nn, activation)

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
    
    def predict(self, sample):
        sample = mx.array(sample)
        logits = self.__call__(sample)
        return nn.softmax(logits, axis=1)
    
    def partials(self, x):
        x = mx.array(x)
        partial = []
        for l in self.layers[:-1]:
            x = self.activation(l(x))
            partial.append(x)
        return np.array(partial)

class TrainingConfig:
    """Configuration class for training parameters."""
    def __init__(
        self,
        num_layers: int = 10,
        hidden_dim: int = 20,
        num_classes: int = 2,
        batch_size: int = 32,
        num_epochs: int = 3000,
        learning_rate: float = 0.0001,
        decay_coeff: float = 0.5,
        activation: str = "relu",
        device: str = "cpu",
        verbose: bool = True
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_coeff = decay_coeff
        self.activation = activation
        self.device = device
        self.verbose = verbose

def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")

def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

def decay_learning_rate(initial_value, decay_coeff, epoch):
    return initial_value/(decay_coeff*(epoch+1))

def get_available_devices() -> Dict[str, Any]:
    """Get information about available devices and their capabilities."""
    devices = {
        'cpu': {
            'count': multiprocessing.cpu_count(),
            'available': True
        },
        'gpu': {
            'count': 0,
            'available': False
        }
    }
    
    # Check for GPU availability
    try:
        if mx.metal.is_available():
            devices['gpu']['available'] = True
            devices['gpu']['count'] = 1  # MLX currently supports one GPU
    except:
        pass
    
    return devices

def calculate_optimal_parallel_models(config_dict: Dict, devices: Dict[str, Any]) -> int:
    """
    Calculate the optimal number of models to train in parallel based on system resources
    and configuration parameters.
    """
    device = config_dict['training']['device'].lower()
    
    if device == 'gpu' and devices['gpu']['available']:
        # For GPU, we'll use a single process but can utilize GPU parallelism
        return 1
    else:
        # For CPU, use process-based parallelism
        cpu_count = devices['cpu']['count']
        total_models = config_dict['training']['parallel']['total_models']
        max_parallel_ratio = config_dict['training']['parallel']['max_parallel_ratio']
        min_parallel = config_dict['training']['parallel']['min_parallel_models']
        max_parallel = config_dict['training']['parallel']['max_parallel_models']
        
        # Calculate optimal number of parallel models
        optimal_parallel = min(
            max_parallel,
            max(
                min_parallel,
                int(cpu_count * max_parallel_ratio)
            )
        )
        
        # Ensure we don't try to run more parallel models than total models
        optimal_parallel = min(optimal_parallel, total_models)
        
        return optimal_parallel

def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config_dict: Dict,
    model_id: int = 0
) -> Tuple[Dict[str, np.ndarray], float, List[float]]:
    """Train a single model instance."""
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
    
    # Set device
    device = config_dict['training']['device'].lower()
    if device == 'gpu' and mx.metal.is_available():
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)
    
    # Convert data to MLX arrays
    X_train = mx.array(X_train)
    if len(y_train.shape) > 1:
        y_train = y_train.reshape(-1)
    y_train = mx.array(y_train.astype(np.int32))
    X_test = mx.array(X_test)
    if len(y_test.shape) > 1:
        y_test = y_test.reshape(-1)
    y_test = mx.array(y_test.astype(np.int32))
    
    # Create model
    model = MLP(
        config.num_layers,
        X_train.shape[-1],
        config.hidden_dim,
        config.num_classes,
        config.activation
    )
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=config.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def step(X, y):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss

    @partial(mx.compile, inputs=model.state)
    def eval_fn(X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)

    accuracies = []
    for e in range(config.num_epochs):
        tic = time.perf_counter()
        optimizer.learning_rate = decay_learning_rate(
            config.learning_rate, 
            config.decay_coeff, 
            e
        )
        
        for X, y in batch_iterate(config.batch_size, X_train, y_train):
            step(X, y)
            mx.eval(model.state)
            
        accuracy = eval_fn(X_test, y_test)
        toc = time.perf_counter()
        
        if e % 10 == 0 and config.verbose:
            print(
                f"Model {model_id} - Epoch {e}: "
                f"Test accuracy {accuracy.item():.3f}, "
                f"Time {toc - tic:.3f} (s)"
            )
        accuracies.append(accuracy.item())

    # Convert model parameters to numpy arrays for pickling
    params = {}
    for k, v in model.parameters().items():
        if isinstance(v, mx.array):
            params[k] = v.numpy()
        elif isinstance(v, list):
            params[k] = [x.numpy() if isinstance(x, mx.array) else x for x in v]
        else:
            params[k] = v
    return params, accuracy.item(), accuracies

def train_parallel(
    X: np.ndarray,
    y: np.ndarray,
    num_models: int = 1,
    config_dict: Dict = None
) -> List[Tuple[Dict[str, np.ndarray], float, List[float]]]:
    """
    Train multiple models in parallel using MLX.
    
    Args:
        X: Input features
        y: Target labels
        num_models: Number of models to train in parallel
        config_dict: Training configuration
        
    Returns:
        List of tuples containing (model_parameters, final_accuracy, accuracy_history)
    """
    if config_dict is None:
        config_dict = yaml.safe_load(open('configs/training_config.yaml', 'r'))

    # Prepare data
    train_images, test_images, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images)
    test_images = scaler.transform(test_images)

    # Get available devices
    devices = get_available_devices()
    device = config_dict['training']['device'].lower()
    
    # Choose appropriate executor based on device
    if device == 'gpu' and devices['gpu']['available']:
        # For GPU, use ThreadPoolExecutor since we're using a single GPU
        executor_class = concurrent.futures.ThreadPoolExecutor
    else:
        # For CPU, use ProcessPoolExecutor for true parallelism
        executor_class = concurrent.futures.ProcessPoolExecutor

    # Train models in parallel
    with executor_class(max_workers=num_models) as executor:
        futures = [
            executor.submit(
                train_single_model,
                train_images,
                train_labels,
                test_images,
                test_labels,
                config_dict,
                i
            )
            for i in range(num_models)
        ]
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return results

def save_mlx_model(params: Dict[str, np.ndarray], path: str) -> None:
    """Save model parameters as a .npz file."""
    np.savez(path, **params)

def train_models_in_batches(
    X: np.ndarray,
    y: np.ndarray,
    config_dict: Dict,
    config: TrainingConfig
) -> List[Tuple[Dict[str, np.ndarray], float, List[float]]]:
    """
    Train models in batches, where each batch runs in parallel.
    """
    # Ensure results/models directory exists
    models_dir = os.path.join('results', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Get available devices
    devices = get_available_devices()
    total_models = config_dict['training']['parallel']['total_models']
    parallel_models = calculate_optimal_parallel_models(config_dict, devices)
    
    print(f"\nSystem Information:")
    print(f"CPU Cores: {devices['cpu']['count']}")
    if devices['gpu']['available']:
        print(f"GPU Available: Yes")
    print(f"Total Models to Train: {total_models}")
    print(f"Models per Batch: {parallel_models}")
    print(f"Number of Batches: {(total_models + parallel_models - 1) // parallel_models}\n")
    
    all_results = []
    for batch_start in range(0, total_models, parallel_models):
        batch_end = min(batch_start + parallel_models, total_models)
        batch_size = batch_end - batch_start
        
        print(f"\nTraining batch {batch_start//parallel_models + 1}: Models {batch_start} to {batch_end-1}")
        
        # Train current batch in parallel
        batch_results = train_parallel(
            X=X,
            y=y,
            num_models=batch_size,
            config_dict=config_dict
        )
        
        all_results.extend(batch_results)
        
        # Save models that meet the threshold
        for i, (params, accuracy, _) in enumerate(batch_results):
            model_idx = batch_start + i
            if accuracy > config_dict['training']['parallel']['save_threshold']:
                model_path = os.path.join(models_dir, f"trained_model_{model_idx}.npz")
                save_mlx_model(params, model_path)
                print(f"Model {model_idx} saved to {model_path}")
    
    return all_results

def validate_config(config_dict: Dict) -> None:
    """Validate the configuration dictionary."""
    required_keys = {
        'model': ['width', 'layers', 'activation'],
        'training': ['epochs', 'batch_size', 'learning_rate', 'device', 'seed', 'parallel'],
        'data': ['test_size', 'random_seed', 'generation']
    }
    
    for section, keys in required_keys.items():
        if section not in config_dict:
            raise ValueError(f"Missing section '{section}' in configuration")
        for key in keys:
            if key not in config_dict[section]:
                raise ValueError(f"Missing key '{key}' in section '{section}'")
    
    # Validate parallel training configuration
    parallel_config = config_dict['training']['parallel']
    if parallel_config['total_models'] < 1:
        raise ValueError("total_models must be at least 1")
    if not 0 < parallel_config['max_parallel_ratio'] <= 1:
        raise ValueError("max_parallel_ratio must be between 0 and 1")
    if parallel_config['min_parallel_models'] < 1:
        raise ValueError("min_parallel_models must be at least 1")
    if parallel_config['max_parallel_models'] < parallel_config['min_parallel_models']:
        raise ValueError("max_parallel_models must be greater than or equal to min_parallel_models")

def main():
    try:
        # Load configuration from YAML
        config_path = 'configs/training_config.yaml'
        print(f"\nLoading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Validate configuration
        print("Validating configuration...")
        validate_config(config_dict)
        print("Configuration validation successful")
        
        # Create training configuration from YAML
        config = TrainingConfig(
            num_layers=config_dict['model']['layers'],
            hidden_dim=config_dict['model']['width'],
            num_classes=2,  # Binary classification for torus data
            batch_size=config_dict['training']['batch_size'],
            num_epochs=config_dict['training']['epochs'],
            learning_rate=config_dict['training']['learning_rate'],
            decay_coeff=config_dict['training']['decay_coeff'],
            activation=config_dict['model']['activation'],
            device=config_dict['training']['device'],
            verbose=True
        )
        
        # Print training configuration
        print("\nTraining Configuration:")
        print(f"Model Architecture: {config_dict['model']['layers']} layers, {config_dict['model']['width']} neurons")
        print(f"Training Parameters: {config_dict['training']['epochs']} epochs, batch size {config_dict['training']['batch_size']}")
        print(f"Learning Rate: {config_dict['training']['learning_rate']} (decay: {config_dict['training']['decay_coeff']})")
        print(f"Device: {config_dict['training']['device']}")
        
        # Generate data using configuration parameters
        print("\nGenerating dataset...")
        X_gen, y_gen = dataset.generate(
            n=config_dict['data']['generation']['n'],
            big_radius=config_dict['data']['generation']['big_radius'],
            small_radius=config_dict['data']['generation']['small_radius']
        )
        print(f"Generated {len(X_gen)} data points")
        
        # Set random seeds for reproducibility
        np.random.seed(config_dict['training']['seed'])
        mx.random.seed(config_dict['training']['seed'])
        print(f"Random seed set to {config_dict['training']['seed']}")
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Train models in batches
        print("\nStarting model training...")
        start = time.time()
        results = train_models_in_batches(X_gen, y_gen, config_dict, config)
        end = time.time()
        
        # Print final results
        print(f"\nTraining completed in {end - start:.2f} seconds")
        print("\nFinal Results:")
        print("-" * 50)
        accuracies = [accuracy for _, accuracy, _ in results]
        print(f"Average Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"Best Accuracy: {max(accuracies):.3f}")
        print(f"Worst Accuracy: {min(accuracies):.3f}")
        print("\nIndividual Model Accuracies:")
        for i, (_, accuracy, _) in enumerate(results):
            print(f"Model {i}: {accuracy:.3f}")
        
        # Save summary to file
        summary_path = os.path.join('results', 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Training Summary\n")
            f.write(f"===============\n\n")
            f.write(f"Configuration:\n")
            f.write(f"- Model: {config_dict['model']['layers']} layers, {config_dict['model']['width']} neurons\n")
            f.write(f"- Training: {config_dict['training']['epochs']} epochs, batch size {config_dict['training']['batch_size']}\n")
            f.write(f"- Learning Rate: {config_dict['training']['learning_rate']}\n\n")
            f.write(f"Results:\n")
            f.write(f"- Training Time: {end - start:.2f} seconds\n")
            f.write(f"- Average Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}\n")
            f.write(f"- Best Accuracy: {max(accuracies):.3f}\n")
            f.write(f"- Worst Accuracy: {min(accuracies):.3f}\n")
        print(f"\nTraining summary saved to {summary_path}")
        
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML configuration: {str(e)}")
        return
    except ValueError as e:
        print(f"Error: Invalid configuration: {str(e)}")
        return
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        return

if __name__ == "__main__":
    main() 