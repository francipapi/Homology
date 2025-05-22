"""
Handles the training of a PyTorch-based neural network model.

This module includes functions to define, train, and evaluate a simple
feed-forward neural network. It processes input data, standardizes features,
trains the model with early stopping, and extracts intermediate ReLU activations
from the trained model. These activations can be used for further analysis,
such as topological data analysis.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time # Not strictly used in the final train function but often useful for debugging/timing
from src.data import dataset # Not strictly used by train() but good for context

def train(X, y, width=32, accuracy=0.2, layers=8, epochs=100, batch_size=32, lr=0.001, device="cpu", seed=None):
    """
    Trains a dense neural network with configurable parameters using PyTorch.

    The network architecture consists of a series of linear layers followed by
    ReLU activations, with a final Sigmoid activation layer for binary
    classification. The training process includes data splitting, feature scaling,
    batched training with an Adam optimizer, and early stopping based on
    validation loss to prevent overfitting. After training, it extracts the
    activations from all ReLU layers for the entire input dataset.

    Args:
        X (np.ndarray): Input features for training and testing.
            Shape should be (n_samples, n_features).
        y (np.ndarray): Target labels for training and testing.
            Shape should be (n_samples,).
        width (int, optional): Number of neurons in each hidden layer.
            Defaults to 32.
        accuracy (float, optional): Desired minimum accuracy on the test set.
            Currently, this parameter is not used to alter the training flow
            (e.g., for early stopping based on target accuracy) but is retained
            for potential future use (e.g., model selection based on this metric).
            Defaults to 0.2.
        layers (int, optional): Number of hidden layers in the network.
            Defaults to 8.
        epochs (int, optional): Maximum number of training epochs.
            Defaults to 100.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        lr (float, optional): Learning rate for the Adam optimizer.
            Defaults to 0.001.
        device (str, optional): Device to use for training. Can be 'cpu',
            'cuda' (for NVIDIA GPUs), or 'mps' (for Apple Silicon GPUs).
            The code attempts to use the specified device, falling back to 'cpu'
            if the requested GPU device is not available. 'auto' is not explicitly
            handled here but by the caller or config (this function expects a specific device string).
            Defaults to "cpu".
        seed (int, optional): Random seed for NumPy and PyTorch to ensure
            reproducibility of weights initialization, data shuffling by PyTorch
            DataLoaders (if `shuffle=True`), and other stochastic processes
            during training. Setting a seed is crucial for comparable results
            across different runs. Defaults to None (no explicit global seed set
            beyond what might be fixed elsewhere, e.g., `random_state` in
            `train_test_split`).

    Returns:
        np.ndarray: A 3D NumPy array containing the activations from all ReLU layers
            for each sample in the input `X` (after scaling). The dimensions are
            (num_relu_layers, num_samples_in_X, layer_width). If no ReLU layers
            are present in the model, an empty NumPy array is returned.
    """
    # Set random seeds for reproducibility if a seed is provided.
    # This affects PyTorch's weight initialization, shuffling in DataLoader, etc.
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Seed all CUDA devices if CUDA is available and intended for use.
        if torch.cuda.is_available() and ('cuda' in device or device == 'auto'):
            torch.cuda.manual_seed_all(seed)
    
    # Determine the effective device for PyTorch operations.
    # Priority: User-specified 'cuda' or 'mps' if available, then 'cpu'.
    # The 'device' variable is updated to reflect the actual device used.
    if device == "auto": # Auto-select device if not specified
        if torch.cuda.is_available():
            effective_device_str = "cuda"
        elif torch.backends.mps.is_available(): # For Apple Silicon GPUs
            effective_device_str = "mps"
        else:
            effective_device_str = "cpu"
    else: # User specified a device
        effective_device_str = device
        # Validate if the requested GPU device is actually available.
        if device == "cuda" and not torch.cuda.is_available():
            print(f"Warning: CUDA device '{device}' requested but not available. Using CPU.")
            effective_device_str = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            print(f"Warning: MPS device '{device}' requested but not available. Using CPU.")
            effective_device_str = "cpu"
        elif device not in ["cpu", "cuda", "mps"]:
            print(f"Warning: Invalid device '{device}' specified. Using CPU.")
            effective_device_str = "cpu"
            
    effective_device = torch.device(effective_device_str)
    print(f"Using device: {effective_device}")

    # Update the 'device' variable to the one that will actually be used.
    # This ensures consistency for the rest of the function.
    device = effective_device

    # --- Data Preprocessing ---
    # Split the dataset into training and testing sets.
    # The random_state is fixed to ensure consistent splits across runs,
    # which is important for comparing results if the main 'seed' changes.
    # test_size determines the proportion of data allocated to the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=52, shuffle=True # test_size & random_state from config?
    )

    # Standardize features by removing the mean and scaling to unit variance.
    # The scaler is fit only on the training data to prevent data leakage from the test set,
    # and then applied to both training and test data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ensure target variables (y_train, y_test) are 1D arrays (squeezed)
    # and cast to float32, which is the expected dtype for PyTorch's BCELoss.
    y_train = y_train.squeeze().astype(np.float32)
    y_test = y_test.squeeze().astype(np.float32) # Ensure y_test is also float32 for consistency

    # --- PyTorch DataLoader Setup ---
    # Convert NumPy arrays to PyTorch tensors and move them to the effective_device.
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create TensorDataset objects for training and testing sets.
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)

    # Create DataLoader instances for managing batches during training and evaluation.
    # shuffle=True for the training loader introduces randomness in batch composition per epoch.
    # shuffle=False for the test loader ensures consistent evaluation.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Definition ---
    # Dynamically construct the neural network as a sequence of layers.
    # Each hidden layer consists of a Linear transformation followed by a ReLU activation.
    # The output layer is a Linear transformation followed by a Sigmoid activation,
    # suitable for binary classification tasks.
    layers_list = []
    input_dim = X.shape[1]  # Input dimension is derived from the number of features in X.

    # Hidden layers
    for _ in range(layers):
        layers_list.append(nn.Linear(input_dim, width))
        layers_list.append(nn.ReLU())
        input_dim = width  # The output dimension of the current layer is the input for the next.

    # Output layer
    layers_list.append(nn.Linear(input_dim, 1)) # Single output neuron for binary classification.
    layers_list.append(nn.Sigmoid()) # Sigmoid activation maps output to a probability (0 to 1).

    # Combine all layers into an nn.Sequential model and move it to the target device.
    model = nn.Sequential(*layers_list).to(device)

    # --- Loss Function and Optimizer ---
    # Binary Cross-Entropy Loss (BCELoss) is used for binary classification.
    criterion = nn.BCELoss()
    # Adam optimizer is chosen for its adaptive learning rate capabilities.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Early Stopping Initialization ---
    # Parameters to control early stopping and keep track of the best model found.
    patience = 75  # Number of epochs to wait for improvement before stopping training.
                   # This value could be made configurable.
    best_val_loss = float('inf')  # Initialize best validation loss to infinity.
    epochs_no_improve = 0  # Counter for epochs without improvement in validation loss.
    best_model_state = None  # Variable to store the state_dict of the best performing model.

    # Training loop with early stopping
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item() * batch_X.size(0)
                preds = (outputs >= 0.5).float()
                correct += (preds.squeeze() == batch_y).sum().item()
                total += batch_X.size(0)
        val_loss /= len(test_loader.dataset)
        val_acc = correct / total

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            model.load_state_dict(best_model_state)
            print("Early Stop Triggered")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, "
                f"Train Loss: {train_loss:.8f}, "
                f"Val Loss: {val_loss:.8f}, "
                f"Val Accuracy: {val_acc:.8f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Function to get ReLU outputs
    def get_relu_outputs(model, X_input):
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
        model.eval()
        relu_activations = []
        with torch.no_grad():
            current = X_tensor
            for layer in model:
                current = layer(current)
                if isinstance(layer, nn.ReLU):
                    relu_activations.append(current.clone())
        return relu_activations

    # Get outputs after ReLU activations
    X_full_scaled = scaler.transform(X)
    layer_output = get_relu_outputs(model, X_full_scaled)

    # Convert list of tensors to a single numpy array
    numpy_list = [tensor.cpu().numpy() for tensor in layer_output]
    numpy_array = np.stack(numpy_list)

    # Print layer shapes
    for i, out in enumerate(layer_output):
        print(f"ReLU Layer {i}: {tuple(out.shape)}")

    return numpy_array