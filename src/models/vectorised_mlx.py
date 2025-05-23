import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, List, Tuple, Any
import yaml
import os
from functools import partial
import time

class VectorizedMLP:
    """A vectorized MLP implementation that trains multiple networks in parallel using MLX's vmap."""
    
    def __init__(
        self,
        num_models: int,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,  # Changed to 1 for binary classification
        activation: str = "relu"
    ):
        """
        Initialize the vectorized MLP.
        
        Args:
            num_models: Number of models to train in parallel
            num_layers: Number of hidden layers
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (default 1 for binary classification)
            activation: Activation function name
        """
        self.num_models = num_models
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = getattr(nn, activation)
        self.training_step = 0  # Initialize training step counter
        
        # Initialize parameters for all models
        self.params = self._init_params()
        
        # Define gradient function
        self._grad_fn = mx.grad(self._forward_single)
        
    def _init_params(self) -> Dict[str, mx.array]:
        """Initialize parameters for all models."""
        params = {}
        layer_sizes = [self.input_dim] + [self.hidden_dim] * self.num_layers + [self.output_dim]
        
        # Initialize weights and biases for each layer
        for i, (idim, odim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            # Initialize weights with Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (idim + odim))
            weights = mx.random.normal((self.num_models, odim, idim)) * scale
            # Initialize biases to small values
            biases = mx.random.normal((self.num_models, odim)) * 0.01
            
            params[f'layer_{i}_weight'] = weights
            params[f'layer_{i}_bias'] = biases
            
        return params
    
    def _forward_single(self, params: Dict[str, mx.array], x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """
        Forward pass for a single model with partial outputs.
        
        Args:
            params: Model parameters
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (final_output, list_of_partial_outputs)
        """
        partial_outputs = []
        for i in range(self.num_layers + 1):
            weight = params[f'layer_{i}_weight']
            bias = params[f'layer_{i}_bias']
            
            # Linear transformation
            x = mx.matmul(x, weight.T) + bias
            
            # Apply activation except for the last layer
            if i < self.num_layers:
                x = self.activation(x)
                partial_outputs.append(x)
            else:
                # Add sigmoid for the last layer
                x = nn.sigmoid(x)
                
        return x, partial_outputs
    
    def forward(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """
        Vectorized forward pass for all models with partial outputs.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (final_output, list_of_partial_outputs)
            final_output shape: (num_models, batch_size, output_dim)
            partial_outputs shape: List of (num_models, batch_size, hidden_dim)
        """
        # Create a vectorized version of the forward pass
        vmap_forward = mx.vmap(
            self._forward_single,
            in_axes=(0, None),  # Parameters are vectorized, input is broadcast
            out_axes=(0, 0)  # Both final output and partial outputs have model dimension first
        )
        
        return vmap_forward(self.params, x)
    
    def partials(self, x: mx.array) -> List[mx.array]:
        """
        Get partial outputs from all layers for all models.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            List of tensors, each with shape (num_models, batch_size, hidden_dim)
        """
        _, partial_outputs = self.forward(x)
        return partial_outputs
    
    def loss_fn(self, logits: mx.array, labels: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Compute binary cross entropy loss for all models.
        
        Args:
            logits: Model outputs of shape (num_models, batch_size, 1)
            labels: True labels of shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Tuple of (total_loss, per_model_losses)
        """
        # Ensure labels are 1D
        if len(labels.shape) > 1:
            labels = labels.reshape(-1)  # Shape: (batch_size,)
            
        # Reshape labels to broadcast with logits
        labels = mx.expand_dims(labels, axis=0)  # Shape: (1, batch_size)
        labels = mx.broadcast_to(labels, (self.num_models, labels.shape[1]))  # Shape: (num_models, batch_size)
        
        # Compute binary cross entropy loss for each model
        per_model_losses = nn.losses.binary_cross_entropy(
            logits.reshape(self.num_models, -1),  # Reshape to (num_models, batch_size)
            labels,
            reduction="none"  # Keep per-sample losses
        )  # Shape: (num_models, batch_size)
        
        # Average over batch dimension
        per_model_losses = mx.mean(per_model_losses, axis=1)  # Shape: (num_models,)
        
        # Total loss is average over all models
        total_loss = mx.mean(per_model_losses)
        
        return total_loss, per_model_losses
    
    def train_step(
        self,
        x: mx.array,
        y: mx.array,
        optimizer: optim.Optimizer
    ) -> Tuple[mx.array, mx.array]:
        """
        Single training step for all models.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            y: Labels of shape (batch_size,)
            optimizer: MLX optimizer
            
        Returns:
            Tuple of (total_loss, per_model_losses)
        """
        start_time = time.time()
        
        def step_fn(params, x, y):
            # Forward pass
            logits, _ = self.forward(x)
            # Compute loss
            total_loss, per_model_losses = self.loss_fn(logits, y)
            return total_loss, per_model_losses
        
        # Get loss and gradients
        loss_and_grad_fn = mx.value_and_grad(step_fn)
        (total_loss, per_model_losses), grads = loss_and_grad_fn(self.params, x, y)
        
        # Update parameters
        optimizer.update(self, grads)
        
        if self.training_step % 100 == 0:
            print(f"Training step took {time.time() - start_time:.4f} seconds")
        
        self.training_step += 1
        return total_loss, per_model_losses
    
    def predict(self, x: mx.array) -> mx.array:
        """
        Get predictions for all models.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (num_models, batch_size, output_dim)
        """
        logits, _ = self.forward(x)
        return nn.sigmoid(logits)
    
    def save_models(self, path: str) -> None:
        """
        Save all model parameters.
        
        Args:
            path: Directory to save models
        """
        os.makedirs(path, exist_ok=True)
        
        # Save each model's parameters separately
        for i in range(self.num_models):
            model_params = {
                k: v[i] for k, v in self.params.items()
            }
            model_path = os.path.join(path, f"model_{i}.npz")
            np.savez(model_path, **{k: np.array(v) for k, v in model_params.items()})

    def full_forward_pass(self, x: mx.array, batch_size: int = 1024) -> Tuple[mx.array, mx.array]:
        """
        Perform forward pass on the entire dataset in batches.
        
        Args:
            x: Full input dataset of shape (dataset_size, input_dim)
            batch_size: Size of batches to process at once
            
        Returns:
            Tuple of (final_outputs, partial_outputs)
            final_outputs shape: (num_models, dataset_size, output_dim)
            partial_outputs shape: (num_models, num_layers, dataset_size, hidden_dim)
        """
        dataset_size = x.shape[0]
        num_batches = (dataset_size + batch_size - 1) // batch_size
        
        # Initialize output arrays
        final_outputs = []
        partial_outputs = [[] for _ in range(self.num_layers)]
        
        # Process in batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, dataset_size)
            batch_x = x[start_idx:end_idx]
            
            # Get outputs for this batch
            batch_final, batch_partials = self.forward(batch_x)
            
            # Store results
            final_outputs.append(batch_final)
            for j, partial in enumerate(batch_partials):
                partial_outputs[j].append(partial)
        
        # Concatenate all batches
        final_outputs = mx.concatenate(final_outputs, axis=1)  # Concatenate along batch dimension
        partial_outputs = [mx.concatenate(p, axis=1) for p in partial_outputs]
        
        # Stack partial outputs along a new dimension
        partial_outputs = mx.stack(partial_outputs, axis=1)  # Shape: (num_models, num_layers, dataset_size, hidden_dim)
        
        return final_outputs, partial_outputs
    
    def get_full_partials(self, x: mx.array, batch_size: int = 1024) -> mx.array:
        """
        Get partial outputs from all layers for all models on the full dataset.
        
        Args:
            x: Full input dataset of shape (dataset_size, input_dim)
            batch_size: Size of batches to process at once
            
        Returns:
            Tensor of shape (num_models, num_layers, dataset_size, hidden_dim)
            containing the outputs of all hidden layers for all models
        """
        _, partial_outputs = self.full_forward_pass(x, batch_size)
        return partial_outputs

    def update(self, new_params):
        """
        Update the model parameters (for optimizer compatibility).
        """
        self.params = new_params

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

def batch_iterate(batch_size, X, y):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

def train_vectorized(
    X: np.ndarray,
    y: np.ndarray,
    config_dict: Dict[str, Any]
) -> Tuple[VectorizedMLP, List[float]]:
    """
    Train multiple models in parallel using vectorized MLX.
    """
    # Convert data to MLX arrays
    X = mx.array(X)
    if len(y.shape) > 1:
        y = y.reshape(-1)
    y = mx.array(y.astype(np.int32))

    # Create vectorized model
    model = VectorizedMLP(
        num_models=config_dict['training']['parallel']['total_models'],
        num_layers=config_dict['model']['layers'],
        input_dim=X.shape[-1],
        hidden_dim=config_dict['model']['width'],
        output_dim=1,  # Binary classification
        activation=config_dict['model']['activation']
    )

    # Create optimizer
    optimizer = optim.Adam(learning_rate=config_dict['training']['learning_rate'])

    batch_size = config_dict['training']['batch_size']
    epochs = config_dict['training']['epochs']
    n = X.shape[0]
    accuracies = []
    X_np = np.array(X)
    y_np = np.array(y)
    for epoch in range(epochs):
        # Training loop over batches
        for batch_X, batch_y in batch_iterate(batch_size, X_np, y_np):
            batch_X_mx = mx.array(batch_X)
            batch_y_mx = mx.array(batch_y)
            model.train_step(batch_X_mx, batch_y_mx, optimizer)
        # Evaluate accuracy on the full dataset
        logits, _ = model.forward(X)
        predictions = (logits.squeeze(-1) > 0.5)
        y_broadcast = mx.broadcast_to(y, (model.num_models, y.shape[0]))
        correct = mx.mean(predictions == y_broadcast, axis=1)
        accuracies.append(np.array(correct))
        # Print progress
        if epoch % 10 == 0:
            total_loss, _ = model.loss_fn(logits, y)
            print(f"Epoch {epoch}:")
            print(f"  Total Loss: {total_loss.item():.4f}")
            print(f"  Average Accuracy: {mx.mean(correct).item():.4f}")
            print(f"  Best Model Accuracy: {mx.max(correct).item():.4f}")
    # Save models that meet the threshold
    threshold = config_dict['training']['parallel']['save_threshold']
    final_accuracies = accuracies[-1]
    if any(acc > threshold for acc in final_accuracies):
        model.save_models(os.path.join('results', 'models'))
    return model, accuracies

def main():
    """Main training function."""
    # Load configuration
    with open('configs/training_config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Generate or load data
    from src.data import dataset
    X, y = dataset.generate(
        n=config_dict['data']['generation']['n'],
        big_radius=config_dict['data']['generation']['big_radius'],
        small_radius=config_dict['data']['generation']['small_radius']
    )
    
    # Train models
    model, accuracies = train_vectorized(X, y, config_dict)
    
    # Save final results
    final_accuracies = accuracies[-1]
    results = {
        'average_accuracy': float(np.mean(final_accuracies)),
        'std_accuracy': float(np.std(final_accuracies)),
        'best_accuracy': float(np.max(final_accuracies)),
        'worst_accuracy': float(np.min(final_accuracies))
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', 'vectorized_training_summary.txt'), 'w') as f:
        f.write("Vectorized Training Results\n")
        f.write("=========================\n\n")
        f.write(f"Average Accuracy: {results['average_accuracy']:.4f} ± {results['std_accuracy']:.4f}\n")
        f.write(f"Best Accuracy: {results['best_accuracy']:.4f}\n")
        f.write(f"Worst Accuracy: {results['worst_accuracy']:.4f}\n")

if __name__ == "__main__":
    main() 