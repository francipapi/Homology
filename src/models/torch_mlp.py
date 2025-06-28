import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import argparse
import trimesh as tr
import os
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training

# --- MLP Class ---
class MLP(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_dim, output_dim, activation_fn_name='relu', dropout_rate=0.2, use_batch_norm=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_fn_name = activation_fn_name.lower()
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Pre-allocate layers for better memory efficiency
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            if self.activation_fn_name == 'relu':
                self.layers.append(nn.ReLU(inplace=True))  # inplace=True for memory efficiency
            elif self.activation_fn_name == 'tanh':
                self.layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {self.activation_fn_name}")

            if self.use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            
            if self.dropout_rate > 0:
                self.layers.append(nn.Dropout(self.dropout_rate))
            
            current_dim = hidden_dim
            
        # Output layer
        self.layers.append(nn.Linear(current_dim, output_dim))
        self.layers.append(nn.Sigmoid())

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity=self.activation_fn_name)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x, extract_hidden_activations=False):
        hidden_activations = []
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        layer_idx = 0
        hidden_layer_count = 0
        
        for layer in self.layers:
            x = layer(x)
            
            if extract_hidden_activations:
                if isinstance(layer, (nn.ReLU, nn.Tanh)):
                    if hidden_layer_count < self.num_hidden_layers:
                        hidden_activations.append(x.detach().clone())
                        hidden_layer_count += 1
            
            layer_idx += 1
        
        if extract_hidden_activations:
            return x, hidden_activations
        return x

    def extract_layer_outputs(self, data_loader, device):
        self.eval()
        
        # Debug print
        print(f"Number of hidden layers: {self.num_hidden_layers}")
        print(f"Using batch norm: {self.use_batch_norm}")
        print(f"Total number of layers: {len(self.layers)}")
        
        # Collect all data
        all_data = []
        for data, _ in data_loader:
            all_data.append(data)
        
        # Concatenate all batches into one tensor
        full_data = torch.cat(all_data, dim=0).to(device)
        print(f"Full dataset shape: {full_data.shape}")
        
        # Process entire dataset at once
        _, activations = self.forward(full_data, extract_hidden_activations=True)
        
        # Debug print
        print(f"Number of activations collected: {len(activations)}")
        for i, act in enumerate(activations):
            print(f"Activation {i} shape: {act.shape}")
        
        if not activations:
            raise RuntimeError("No activations were collected. Check the model architecture and activation collection logic.")
        
        # Stack to (num_hidden_layers, dataset_size, hidden_dimension)
        stacked_activations = torch.stack(activations, dim=0)  # (num_hidden_layers, dataset_size, hidden_dim)
        
        # Add leading batch dimension: (1, num_hidden_layers, dataset_size, hidden_dim)
        output_tensor = stacked_activations.unsqueeze(0)
        return output_tensor


# --- Data Loading Functions ---
def load_data_from_file(file_path):
    """
    Load dataset from a file. Supports .npy, .npz, .pt, and .pth formats.
    Expected format: X (features) and y (labels) arrays.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if file_path.suffix == '.npy':
        # Assume the file contains both X and y in a single array or dict
        data = np.load(file_path, allow_pickle=True)
        if isinstance(data, dict) or hasattr(data, 'item'):
            data = data.item() if hasattr(data, 'item') else data
            X = torch.tensor(data['X'], dtype=torch.float32)
            y = torch.tensor(data['y'], dtype=torch.float32)
        else:
            raise ValueError("For .npy files, expected dict with 'X' and 'y' keys")
    elif file_path.suffix == '.npz':
        data = np.load(file_path)
        X = torch.tensor(data['X'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.float32)
    elif file_path.suffix in ['.pt', '.pth']:
        data = torch.load(file_path)
        if isinstance(data, dict):
            X = data['X']
            y = data['y']
        else:
            raise ValueError("For .pt/.pth files, expected dict with 'X' and 'y' keys")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return X, y

# --- Data Generation ---
def generate_torus_data(n_samples, big_radius, small_radius, solid=False, interior_noise=0.1):
    """
    Generate torus data using the centralized dataset generation functions.
    
    Parameters:
    - n_samples: Number of points per torus
    - big_radius: Major radius of torus
    - small_radius: Minor radius of torus  
    - solid: If True, generate solid tori; if False, hollow (surface-only)
    - interior_noise: Noise level for interior points when solid=True
    
    Returns:
    - X, y: PyTorch tensors with point cloud data and labels
    """
    # Import the dataset generation functions
    import sys
    import os
    # Add project root to Python path if not already there
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.data.dataset import generate
    
    # Use the centralized generation function
    X, y = generate(n_samples, big_radius, small_radius, solid, interior_noise)
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    return X, y

# --- Training Function ---
def train_model(config_path):
    # Load and clean config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    training_config = config['training']
    data_config = config['data']

    # Device setup with fallback
    if training_config['device'] == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(training_config['device'])
    
    print(f"Using device: {device}")

    # Model
    model = MLP(
        input_dim=model_config['input_dim'],
        num_hidden_layers=model_config['num_hidden_layers'],
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        activation_fn_name=model_config.get('activation_fn_name', 'relu'),
        dropout_rate=model_config.get('dropout_rate', 0.0),
        use_batch_norm=model_config.get('use_batch_norm', False)
    ).to(device)

    # Enable cuDNN benchmarking for faster training
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Optimizer
    lr = training_config['learning_rate']
    opt_config = training_config.get('optimizer', {'type': 'adam'})
    optimizer_type = opt_config.get('type', 'adam').lower()
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0.0))
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0.01))
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0.0))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    # Get regularization parameters
    reg_config = training_config.get('regularization', {})
    l1_lambda = reg_config.get('l1_lambda', 0.0)
    l2_lambda = reg_config.get('l2_lambda', 0.0)

    # Loss function
    criterion = nn.BCELoss()

    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None

    # Data generation or loading
    data_source = data_config.get('data_source')
    if data_source is not None:
        print(f"Loading data from: {data_source}")
        X, y = load_data_from_file(data_source)
    elif data_config['type'] == 'synthetic':
        num_samples = data_config.get('generation', {}).get('n', 1000)
        big_radius = data_config.get('generation', {}).get('big_radius', 3)
        small_radius = data_config.get('generation', {}).get('small_radius', 1)
        solid = data_config.get('generation', {}).get('solid', False)
        interior_noise = data_config.get('generation', {}).get('interior_noise', 0.1)
        X, y = generate_torus_data(num_samples, big_radius, small_radius, solid, interior_noise)
    else:
        raise ValueError(f"Unsupported data configuration. Either set data_source or use synthetic data.")
        
    # Move data to device
    X = X.to(device)
    y = y.to(device)

    # Shiffle data 
    perm = torch.randperm(len(X), device=device)   # random index order
    X = X[perm]
    y = y[perm]
    
    # Split data
    split_ratio = data_config.get('split_ratio', 0.8)
    train_size = int(split_ratio * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], 
                            shuffle=True, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=training_config['batch_size'], 
                           shuffle=False, pin_memory=True if device.type == 'cuda' else False)

    # Scheduler
    scheduler = None
    scheduler_config = training_config.get('scheduler', {})
    if scheduler_config.get('type') == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=scheduler_config.get('factor', 0.1), 
            patience=scheduler_config.get('patience', 10)
        )

    # Training Loop
    for epoch in range(training_config['epochs']):
        model.train()
        train_loss_sum = 0
        correct_train = 0
        total_train = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler is not None:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Add L1/L2 regularization
                    if l1_lambda > 0 or l2_lambda > 0:
                        reg_loss = 0
                        for param in model.parameters():
                            if l1_lambda > 0:
                                reg_loss += l1_lambda * torch.sum(torch.abs(param))
                            if l2_lambda > 0:
                                reg_loss += l2_lambda * torch.sum(param ** 2)
                        loss = loss + reg_loss
                        
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                
                # Add L1/L2 regularization
                if l1_lambda > 0 or l2_lambda > 0:
                    reg_loss = 0
                    for param in model.parameters():
                        if l1_lambda > 0:
                            reg_loss += l1_lambda * torch.sum(torch.abs(param))
                        if l2_lambda > 0:
                            reg_loss += l2_lambda * torch.sum(param ** 2)
                    loss = loss + reg_loss
                    
                loss.backward()
                optimizer.step()

            train_loss_sum += loss.item()
            predicted = (output > 0.5).float()
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        train_accuracy = correct_train / total_train

        # Evaluation
        model.eval()
        test_loss_sum = 0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                test_loss_sum += loss.item()
                
                predicted = (output > 0.5).float()
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()

        avg_test_loss = test_loss_sum / len(test_loader)
        test_accuracy = correct_test / total_test

        print(f"Epoch {epoch+1}/{training_config['epochs']} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

        if scheduler:
            scheduler.step(avg_test_loss)
    
    print("Training finished.")

    # Extract layer outputs if enabled
    layer_extraction_config = config.get('layer_extraction', {})
    if layer_extraction_config.get('enabled', False):
        print("\nExtracting layer outputs...")
        model.to(device)
        # Combine train and test datasets into one
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        full_loader = DataLoader(
            full_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            pin_memory=True if device.type == 'cuda' else False
        )
        layer_outputs_tensor = model.extract_layer_outputs(full_loader, device)
        print(f"torch_mlp.py: Shape of extracted layer outputs tensor: {layer_outputs_tensor.shape}")
        
        # Save layer outputs
        output_dir = Path(layer_extraction_config.get('output_dir', 'results/layer_outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'torch_mlp_layer_outputs.pt'
        torch.save({
            'layer_outputs': layer_outputs_tensor.cpu(),
            'config': config
        }, output_file)
        print(f"Layer outputs saved to: {output_file}")
    else:
        print("Layer extraction disabled. Skipping layer output extraction.")
    
    # Extract train/test activations separately if enabled
    if layer_extraction_config.get('train_test_layer_extraction', False):
        print("\nExtracting train and test layer outputs separately...")
        
        # Create output directory for train/test outputs
        train_test_output_dir = Path(layer_extraction_config.get('train_test_output_dir', 'results/train_test_layer_outputs'))
        train_test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract train dataset activations
        print("Extracting train dataset activations...")
        train_layer_outputs = model.extract_layer_outputs(train_loader, device)
        train_output_file = train_test_output_dir / 'torch_mlp_train_layer_outputs.pt'
        torch.save({
            'layer_outputs': train_layer_outputs.cpu(),
            'dataset_type': 'train',
            'dataset_size': len(train_dataset),
            'config': config
        }, train_output_file)
        print(f"Train layer outputs saved to: {train_output_file}")
        
        # Extract test dataset activations
        print("Extracting test dataset activations...")
        test_layer_outputs = model.extract_layer_outputs(test_loader, device)
        test_output_file = train_test_output_dir / 'torch_mlp_test_layer_outputs.pt'
        torch.save({
            'layer_outputs': test_layer_outputs.cpu(),
            'dataset_type': 'test',
            'dataset_size': len(test_dataset),
            'config': config
        }, test_output_file)
        print(f"Test layer outputs saved to: {test_output_file}")
        
        print(f"Train/test layer extraction complete. Shape: {train_layer_outputs.shape}")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP model using a YAML configuration file.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    train_model(args.config_path)