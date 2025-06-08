import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data import dataset

def parallel_train(param):
    X, y = param
    width=15
    accuracy=0.2
    layers=10
    epochs=1000
    batch_size=32
    lr=0.001
    device="cpu"
    activation="tanh"

    # Move to the selected device
    device = torch.device(device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=52, shuffle=True
    )

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ensure y is (N,) to avoid shape mismatch
    y_train = y_train.squeeze().astype(np.float32)
    y_test = y_test.squeeze()

    # Convert to torch tensors and move to device
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define activation functions mapping
    activation_functions = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leakyrelu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        # Add more activation functions here if needed
    }

    activation_functions_names = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leakyrelu': nn.LeakyReLU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
        # Add more activation functions here if needed
    }

    # Validate activation function
    activation = activation.lower()
    if activation not in activation_functions:
        raise ValueError(f"Unsupported activation function '{activation}'. Supported options are: {list(activation_functions.keys())}")

    activation_fn = activation_functions[activation]

    # Define the model
    layers_list = []
    input_dim = X.shape[1]

    for _ in range(layers):
        layers_list.append(nn.Linear(input_dim, width))
        layers_list.append(activation_fn)
        input_dim = width

    layers_list.append(nn.Linear(input_dim, 1))
    layers_list.append(nn.Sigmoid())  # Output layer activation remains Sigmoid for binary classification

    model = nn.Sequential(*layers_list).to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.8,0.99))

    # Early stopping parameters
    patience = 75
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

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

    # Function to get activation outputs
    def get_activation_outputs(model, X_input, activation_name):
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
        model.eval()
        activation_activations = []
        with torch.no_grad():
            current = X_tensor
            for layer in model:
                current = layer(current)
                if isinstance(layer, activation_functions_names[activation_name]):
                    activation_activations.append(current.clone())
        return activation_activations

    # Get outputs after specified activations
    X_full_scaled = scaler.transform(X)
    layer_output = get_activation_outputs(model, X_full_scaled, activation)

    # Convert list of tensors to a single numpy array
    numpy_list = [tensor.cpu().numpy() for tensor in layer_output]
    numpy_array = np.stack(numpy_list)

    # Print layer shapes
    for i, out in enumerate(layer_output):
        print(f"{activation.capitalize()} Layer {i+1}: {tuple(out.shape)}")

    return numpy_array

def train(X, y, width=32, accuracy=0.2, layers=8, epochs=100, batch_size=32, lr=0.001, device="cpu", activation="relu"):

    """
    Train a dense neural network with configurable parameters using PyTorch.

    Parameters:
    - X (np.ndarray): Input features.
    - y (np.ndarray): Target labels.
    - width (int): Number of neurons per hidden layer.
    - accuracy (float): Desired minimum accuracy.
    - layers (int): Number of dense hidden layers.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - lr (float): Learning rate.
    - device (str): 'cpu' or 'mps' for Apple M1/M2 GPUs, or 'cuda' for NVIDIA GPUs.
    - activation (str): Activation function to use in hidden layers. Options include
                        'relu', 'tanh', 'sigmoid', 'leakyrelu', etc.

    Returns:
    - numpy_array (np.ndarray): Outputs after the specified activations for the entire dataset X.
    """
    # Set random seeds for reproducibility
    # seed = 52
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    
    # Move to the selected device
    device = torch.device(device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=52, shuffle=True
    )

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ensure y is (N,) to avoid shape mismatch
    y_train = y_train.squeeze().astype(np.float32)
    y_test = y_test.squeeze()

    # Convert to torch tensors and move to device
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define activation functions mapping
    activation_functions = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leakyrelu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        # Add more activation functions here if needed
    }

    # Validate activation function
    activation = activation.lower()
    if activation not in activation_functions:
        raise ValueError(f"Unsupported activation function '{activation}'. Supported options are: {list(activation_functions.keys())}")

    activation_fn = activation_functions[activation]

    # Define the model
    layers_list = []
    input_dim = X.shape[1]

    for _ in range(layers):
        layers_list.append(nn.Linear(input_dim, width))
        layers_list.append(activation_fn)
        input_dim = width

    layers_list.append(nn.Linear(input_dim, 1))
    layers_list.append(nn.Sigmoid())  # Output layer activation remains Sigmoid for binary classification

    model = nn.Sequential(*layers_list).to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early stopping parameters
    patience = 75
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

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

    # Function to get activation outputs
    def get_activation_outputs(model, X_input, activation_name):
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
        model.eval()
        activation_activations = []
        with torch.no_grad():
            current = X_tensor
            for layer in model:
                current = layer(current)
                if isinstance(layer, activation_functions[activation_name]):
                    activation_activations.append(current.clone())
        return activation_activations

    # Get outputs after specified activations
    X_full_scaled = scaler.transform(X)
    layer_output = get_activation_outputs(model, X_full_scaled, activation)

    # Convert list of tensors to a single numpy array
    numpy_list = [tensor.cpu().numpy() for tensor in layer_output]
    numpy_array = np.stack(numpy_list)

    # Print layer shapes
    for i, out in enumerate(layer_output):
        print(f"{activation.capitalize()} Layer {i+1}: {tuple(out.shape)}")

    return numpy_array

def parallel_train_tanh(param):
    X, y = param
    width=20
    accuracy=0.2
    layers=10
    epochs=300
    batch_size=128
    lr=0.001
    device="cpu"

    # Move to the selected device
    device = torch.device(device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=52, shuffle=True
    )

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ensure y is (N,) to avoid shape mismatch
    y_train = y_train.squeeze().astype(np.float32)
    y_test = y_test.squeeze()

    # Convert to torch tensors and move to device
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    layers_list = []
    input_dim = X.shape[1]

    for _ in range(layers):
        layers_list.append(nn.Linear(input_dim, width))
        layers_list.append(nn.Tanh())
        input_dim = width

    layers_list.append(nn.Linear(input_dim, 1))
    layers_list.append(nn.Sigmoid())

    model = nn.Sequential(*layers_list).to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early stopping parameters
    patience = 75
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

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
                if isinstance(layer, nn.Tanh):
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
'''
def main():
    X, y = data.generate(8000,3,1)
    parallel_train_tanh([X,y])
    
if __name__ == "__main__":
    main()
'''