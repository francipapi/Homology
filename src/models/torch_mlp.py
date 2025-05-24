import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import argparse
import trimesh as tr
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


# --- Data Generation ---
def generate_torus_data(n_samples, big_radius, small_radius):
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
        points = np.array(torus.sample(n_samples))
        sampled_points.append(points)
        labels.append(np.full((n_samples, 1), i % 2))  # Alternating labels 0 and 1
    
    # Concatenate results
    X = np.concatenate(sampled_points)
    y = np.concatenate(labels)
    
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

    # Loss function
    criterion = nn.BCELoss()

    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None

    # Data generation
    if data_config['type'] == 'synthetic':
        num_samples = data_config.get('generation', {}).get('n', 1000)
        big_radius = data_config.get('generation', {}).get('big_radius', 3)
        small_radius = data_config.get('generation', {}).get('small_radius', 1)
        X, y = generate_torus_data(num_samples, big_radius, small_radius)
        
        # Move data to device
        X = X.to(device)
        y = y.to(device)
        
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
    else:
        raise ValueError(f"Unsupported data type: {data_config['type']}")

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
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
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

    # Extract layer outputs from the *full* dataset
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

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP model using a YAML configuration file.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    train_model(args.config_path)
