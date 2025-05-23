import sys
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from sklearn.datasets import make_moons
import trimesh as tr
import numpy as np
import argparse

# Copied generate function from src/data/dataset.py
def generate(n, big_radius, small_radius):
    # Helper function for creating and transforming a torus pair
    def create_transformed_torus_pair(offset, rotation_axis, rotation_angle, translation_vector):
        torus1 = tr.creation.torus(big_radius, small_radius)
        torus2 = tr.creation.torus(big_radius, small_radius)
        
        # Apply rotation to the second torus
        rotation_matrix = tr.transformations.rotation_matrix(rotation_angle, rotation_axis)
        torus2.apply_transform(rotation_matrix)
        
        # Apply translation to separate the tori
        translation_matrix1 = tr.transformations.translation_matrix([big_radius/2, 0, 0])
        translation_matrix2 = tr.transformations.translation_matrix([-big_radius/2, 0, 0])
        torus2.apply_transform(translation_matrix2)
        torus1.apply_transform(translation_matrix1)
        
        # Apply offsets for positioning
        torus1.apply_transform(tr.transformations.translation_matrix(translation_vector))
        torus2.apply_transform(tr.transformations.translation_matrix(translation_vector))
        
        return torus1, torus2

    # Define translations
    scale_factor = big_radius * 3
    translations = [
        [-scale_factor, scale_factor, scale_factor],
        [-scale_factor, -scale_factor, scale_factor],
        [-scale_factor, scale_factor, -scale_factor],
        [-scale_factor, -scale_factor, -scale_factor],
        [scale_factor, scale_factor, scale_factor],
        [scale_factor, -scale_factor, scale_factor],
        [scale_factor, scale_factor, -scale_factor],
        [scale_factor, -scale_factor, -scale_factor]
    ]
    
    # Create tori pairs with transformations
    torus_pairs = []
    for translation in translations:
        torus1, torus2 = create_transformed_torus_pair(
            offset=big_radius, 
            rotation_axis=[1, 0, 0], 
            rotation_angle=np.pi / 2, 
            translation_vector=translation
        )
        torus_pairs.extend([torus1, torus2])
    
    # Sample points from all the tori
    sampled_points = []
    labels = []
    for i, torus in enumerate(torus_pairs):
        points = np.array(torus.sample(n))
        sampled_points.append(points)
        labels.append(np.full((n, 1), i % 2))  # Alternating labels 0 and 1
    
    # Concatenate results
    X = np.concatenate(sampled_points)
    y = np.concatenate(labels)
    
    return [X, y]

class SimpleMLP:
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        self.layers = []
        self.activation = activation

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
        self.layers.append(nn.Dropout(0.4))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            self.layers.append(nn.Dropout(0.4))

        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers.append(nn.Sigmoid())

        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # He initialization
                scale = float(mx.sqrt(2.0 / layer.weight.shape[0]).item())
                layer.weight = mx.random.normal(list(layer.weight.shape), loc=0, scale=scale)
                layer.bias = mx.zeros(list(layer.bias.shape))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                params[f'layer{i}'] = {
                    'weight': layer.weight,
                    'bias': layer.bias
                }
        return params

    def update(self, new_params):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                layer.weight = new_params[f'layer{i}']['weight']
                layer.bias = new_params[f'layer{i}']['bias']

def binary_cross_entropy(pred, target):
    # pred: (batch, 1), target: (batch, 1)
    return -mx.mean(target * mx.log(pred + 1e-8) + (1 - target) * mx.log(1 - pred + 1e-8))

def loss_fn(params, model, X, y):
    # Use the same architecture as the main model
    input_dim = model.layers[0].weight.shape[1]
    hidden_dims = [layer.weight.shape[1] for layer in model.layers if isinstance(layer, nn.Linear)][:-1]
    temp_model = SimpleMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)
    for i, layer in enumerate(temp_model.layers):
        if isinstance(layer, nn.Linear):
            layer.weight = params[f'layer{i}']['weight']
            layer.bias = params[f'layer{i}']['bias']
    logits = temp_model(X)
    loss = binary_cross_entropy(logits, y)
    return loss

def main():
    # Add argument parser for device selection
    parser = argparse.ArgumentParser(description='Train MLP on synthetic data')
    parser.add_argument('--torus', action='store_true', help='Use torus dataset instead of moons')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], 
                       help='Device to use for training (cpu or gpu)')
    args = parser.parse_args()

    # Set default device
    mx.set_default_device(mx.gpu if args.device == 'gpu' else mx.cpu)
    print(f"Using device: {args.device}")

    use_torus = args.torus
    if use_torus:
        print("Using torus synthetic dataset from dataset.py ...")
        X, y = generate(4000, 3, 1)
        # X: (N, 3), y: (N, 1)
        input_dim = 3
        y = y.reshape(-1, 1)  # Ensure shape (N, 1)
    else:
        print("Using sklearn make_moons dataset ...")
        X, y = make_moons(n_samples=4000, noise=0.1, random_state=42)
        input_dim = 2
        y = y.reshape(-1, 1)
    
    # Create arrays (they will automatically use the default device)
    X = mx.array(X, dtype=mx.float32)
    y = mx.array(y, dtype=mx.float32)

    # Shuffle the dataset before splitting
    indices = mx.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize model with adjusted architecture
    model = SimpleMLP(
        input_dim=input_dim,
        hidden_dims=[256, 256, 128, 128, 64, 64, 32, 32],
        output_dim=1
    )
    
    # Initialize optimizer with higher learning rate
    optimizer = optim.Adam(
        learning_rate=0.05,
        eps=1e-8
    )

    # Training loop
    batch_size = 128
    n_epochs = 40

    # Learning rate scheduler
    initial_lr = optimizer.learning_rate
    for epoch in range(n_epochs):
        # Adjust learning rate
        if epoch > 0 and epoch % 5 == 0:
            optimizer.learning_rate = initial_lr * (0.1 ** (epoch // 5))
            print(f"Reducing learning rate to: {optimizer.learning_rate}")

        # Shuffle training data
        indices = mx.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Training
        train_loss = 0
        train_correct = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            # Compute loss and gradients
            params = model.parameters()
            loss, grads = mx.value_and_grad(lambda p: loss_fn(p, model, batch_X, batch_y))(params)
            train_loss += loss.item()

            # Update parameters
            new_params = tree_map(lambda g, p: p - optimizer.learning_rate * g, grads, params)
            model.update(new_params)

            # Compute accuracy
            preds = mx.stop_gradient((model(batch_X) > 0.5).astype(mx.float32))
            train_correct += mx.sum(preds == batch_y).item()

            # Print predictions and targets for the first batch of the first epoch
            if epoch == 0 and i == 0:
                print('First batch predictions:', model(batch_X)[:10].tolist())
                print('First batch targets:', batch_y[:10].tolist())

        # Evaluation
        test_preds = mx.stop_gradient((model(X_test) > 0.5).astype(mx.float32))
        test_loss = loss_fn(model.parameters(), model, X_test, y_test).item()
        test_correct = mx.sum(test_preds == y_test).item()

        # Print metrics
        train_loss /= len(X_train) // batch_size
        train_acc = (train_correct / len(X_train)) * 100  # Convert to percentage
        test_acc = (test_correct / len(X_test)) * 100  # Convert to percentage
        print(f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main() 