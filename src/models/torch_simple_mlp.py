import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from sklearn.datasets import make_moons
import trimesh as tr

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

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
        self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.layers.append(nn.Dropout(0.2))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            self.layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.layers.append(nn.Dropout(0.2))

        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers.append(nn.Sigmoid())

        # Initialize weights using He initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu' if activation == 'relu' else 'tanh')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def main():
    # Add argument parser for device selection
    parser = argparse.ArgumentParser(description='Train MLP on synthetic data')
    parser.add_argument('--torus', action='store_true', help='Use torus dataset instead of moons')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                       help='Device to use for training (cpu or cuda)')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
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
    
    # Convert to PyTorch tensors and move to device
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    # Shuffle the dataset before splitting
    indices = torch.randperm(len(X))
    X = X[indices]
    y = y[indices]

    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize model with a more efficient architecture
    model = SimpleMLP(
        input_dim=input_dim,
        hidden_dims=[32, 32, 32, 32, 32, 32, 32, 32],
        output_dim=1
    ).to(device)
    
    # Initialize optimizer with weight decay and higher learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.01,
        weight_decay=0.001,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Training loop
    batch_size = 128
    n_epochs = 100
    criterion = nn.BCELoss()
    max_grad_norm = 1.0  # For gradient clipping

    best_test_acc = 0
    patience = 20
    patience_counter = 0

    for epoch in range(n_epochs):
        # Shuffle training data
        indices = torch.randperm(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            train_loss += loss.item()

            # Compute accuracy
            preds = (outputs > 0.5).float()
            train_correct += (preds == batch_y).sum().item()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_preds = (test_outputs > 0.5).float()
            test_correct = (test_preds == y_test).sum().item()

        # Update learning rate based on validation loss
        scheduler.step(test_loss)

        # Print metrics
        train_loss /= len(X_train) // batch_size
        train_acc = (train_correct / len(X_train)) * 100
        test_acc = (test_correct / len(X_test)) * 100
        print(f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

if __name__ == "__main__":
    main() 