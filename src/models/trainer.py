import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from src.data import dataset

def parallel_train(param):
    X, y = param
    width=15
    accuracy=0.2
    layers=10
    epochs=100
    batch_size=32
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
        layers_list.append(nn.ReLU())
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

def train(X, y, width=32, accuracy=0.2, layers=8, epochs=100, batch_size=32, lr=0.001, device="cpu"):
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

    Returns:
    - numpy_array (np.ndarray): Outputs after ReLU activations for the entire dataset X.
    """
    '''
    # Set random seeds for reproducibility
    seed = 52
    np.random.seed(seed)
    torch.manual_seed(seed)
    '''
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
        layers_list.append(nn.ReLU())
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

'''

X_gen, y_gen = data.generate(4000, 3, 1)
start = time.time()
print(parallel_train([X_gen, y_gen]).shape)
end = time.time()
print(end-start)

'''




'''
TENSORFLOW VERSION

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from tensorflow.keras.callbacks import EarlyStopping

def train(X, y, width=32, accuracy=0.2, layers=8, epochs=100, batch_size=32, lr=0.001):
    """
    Train a dense neural network with configurable parameters.

    Parameters:
    - X (array): Input features.
    - y (array): Target labels.
    - width (int): Number of neurons per layer.
    - accuracy (float): Desired minimum accuracy.
    - layers (int): Number of dense layers in the model.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - layer_outputs (list): Outputs of each layer for the entire dataset X.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52, shuffle=True)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Build the model
    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    for _ in range(layers):
        model.add(Dense(width, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    if test_acc < accuracy:
        print(f"Model accuracy {test_acc:.4f} did not meet desired threshold {accuracy:.4f}")
        return 0

    def getLayerOut(m,i):
            output = []
            x = i
            for layer in model.layers:
                x = layer(x)
                output.append(x)
            return output

    layer_output = getLayerOut(model,X)
    for i, o in enumerate(layer_output):
        print(f"Layer {i}: {o.shape}")
        
    return layer_output


def Train(X,y, width=8, accuracy=0.55):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52, shuffle=True)
    scaler = StandardScaler()
    scaler.fit(X)
    X_test = scaler.transform(X_test)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    model = Sequential([
    Input(shape=(3,)),
    Dense(width, activation='relu'),
    Dense(width, activation='relu'),
    Dense(width, activation='relu'),
    Dense(width, activation='relu'),
    Dense(width, activation='relu'),
    Dense(width, activation='relu'),
    Dense(width, activation='relu'),
    Dense(1, activation='sigmoid')])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    if model.evaluate(X_test, y_test)[1] < accuracy:
        return 0

    def getLayerOut(m,i):
        output = []
        x = i
        for layer in model.layers:
            x = layer(x)
            output.append(x)
        return output

    layer_output = getLayerOut(model,X)
    for i, o in enumerate(layer_output):
        print(f"Layer {i}: {o.shape}")
    
    return layer_output
'''