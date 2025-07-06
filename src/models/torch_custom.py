import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from typing import List, Dict, Any, Tuple, Optional, Union
import sys
import os

# Enable MPS fallback for operations not supported natively on MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.torch_mlp import generate_torus_data, load_data_from_file


class CustomNet(nn.Module):
    """
    A flexible neural network that can be configured with mixed layer types
    including convolutional and linear layers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(CustomNet, self).__init__()
        
        self.config = config
        self.input_shape = config['input_shape']
        self.layers_config = config['layers']
        self.extract_from_layers = config.get('extract_from_layers', 'all')
        self.flatten_conv_activations = config.get('flatten_conv_activations', True)
        self.skip_final_layer = config.get('skip_final_layer', True)  # Skip extracting final output layer
        
        # Build the network
        self.layers = nn.ModuleList()
        self.layer_types = []  # Track layer types for extraction
        self.activation_indices = []  # Track which layers have activations to extract
        
        self._build_network()
        self._initialize_weights()
    
    def _get_activation_fn(self, name: str) -> Optional[nn.Module]:
        """Get activation function by name."""
        if name is None or name.lower() == 'none':
            return None
        
        activations = {
            'relu': nn.ReLU(inplace=True),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=1),
            'leaky_relu': nn.LeakyReLU(inplace=True),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU()
        }
        
        name_lower = name.lower()
        if name_lower not in activations:
            raise ValueError(f"Unsupported activation function: {name}")
        
        return activations[name_lower]
    
    def _build_network(self):
        """Build the network from configuration."""
        current_shape = list(self.input_shape)
        
        for i, layer_config in enumerate(self.layers_config):
            layer_type = layer_config['type'].lower()
            
            if layer_type == 'linear':
                # Flatten if needed before linear layer
                if len(current_shape) > 1:
                    self.layers.append(nn.Flatten())
                    self.layer_types.append('flatten')
                    current_shape = [np.prod(current_shape)]
                
                out_features = layer_config['out_features']
                self.layers.append(nn.Linear(current_shape[0], out_features))
                self.layer_types.append('linear')
                current_shape = [out_features]
                
            elif layer_type == 'conv1d':
                if len(current_shape) == 1:
                    raise ValueError(f"Conv1d requires 2D input [channels, length], got shape {current_shape}")
                
                out_channels = layer_config['out_channels']
                kernel_size = layer_config.get('kernel_size', 1)
                stride = layer_config.get('stride', 1)
                padding = layer_config.get('padding', 0)
                
                in_channels = current_shape[0]
                self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
                self.layer_types.append('conv1d')
                
                # Update shape
                length = current_shape[1]
                new_length = (length + 2 * padding - kernel_size) // stride + 1
                current_shape = [out_channels, new_length]
                
            elif layer_type == 'conv2d':
                if len(current_shape) < 3:
                    raise ValueError(f"Conv2d requires 3D input [channels, height, width], got shape {current_shape}")
                
                out_channels = layer_config['out_channels']
                kernel_size = layer_config.get('kernel_size', 3)
                stride = layer_config.get('stride', 1)
                padding = layer_config.get('padding', 0)
                
                in_channels = current_shape[0]
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
                self.layer_types.append('conv2d')
                
                # Update shape
                height, width = current_shape[1], current_shape[2]
                new_height = (height + 2 * padding - kernel_size) // stride + 1
                new_width = (width + 2 * padding - kernel_size) // stride + 1
                current_shape = [out_channels, new_height, new_width]
                
            elif layer_type == 'maxpool1d':
                kernel_size = layer_config.get('kernel_size', 2)
                stride = layer_config.get('stride', kernel_size)
                padding = layer_config.get('padding', 0)
                
                self.layers.append(nn.MaxPool1d(kernel_size, stride, padding))
                self.layer_types.append('maxpool1d')
                
                # Update shape
                length = current_shape[1]
                new_length = (length + 2 * padding - kernel_size) // stride + 1
                current_shape[1] = new_length
                
            elif layer_type == 'maxpool2d':
                kernel_size = layer_config.get('kernel_size', 2)
                stride = layer_config.get('stride', kernel_size)
                padding = layer_config.get('padding', 0)
                
                self.layers.append(nn.MaxPool2d(kernel_size, stride, padding))
                self.layer_types.append('maxpool2d')
                
                # Update shape
                height, width = current_shape[1], current_shape[2]
                new_height = (height + 2 * padding - kernel_size) // stride + 1
                new_width = (width + 2 * padding - kernel_size) // stride + 1
                current_shape[1], current_shape[2] = new_height, new_width
                
            elif layer_type == 'avgpool1d':
                kernel_size = layer_config.get('kernel_size', 2)
                stride = layer_config.get('stride', kernel_size)
                padding = layer_config.get('padding', 0)
                
                self.layers.append(nn.AvgPool1d(kernel_size, stride, padding))
                self.layer_types.append('avgpool1d')
                
                # Update shape
                length = current_shape[1]
                new_length = (length + 2 * padding - kernel_size) // stride + 1
                current_shape[1] = new_length
                
            elif layer_type == 'avgpool2d':
                kernel_size = layer_config.get('kernel_size', 2)
                stride = layer_config.get('stride', kernel_size)
                padding = layer_config.get('padding', 0)
                
                self.layers.append(nn.AvgPool2d(kernel_size, stride, padding))
                self.layer_types.append('avgpool2d')
                
                # Update shape
                height, width = current_shape[1], current_shape[2]
                new_height = (height + 2 * padding - kernel_size) // stride + 1
                new_width = (width + 2 * padding - kernel_size) // stride + 1
                current_shape[1], current_shape[2] = new_height, new_width
                
            elif layer_type == 'flatten':
                self.layers.append(nn.Flatten())
                self.layer_types.append('flatten')
                current_shape = [np.prod(current_shape)]
                
            elif layer_type == 'reshape':
                target_shape = layer_config['shape']
                self.layers.append(Reshape(target_shape))
                self.layer_types.append('reshape')
                current_shape = list(target_shape)
                
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            
            # Add activation function if specified
            activation_name = layer_config.get('activation')
            activation_fn = self._get_activation_fn(activation_name)
            if activation_fn is not None:
                self.layers.append(activation_fn)
                self.layer_types.append(f'activation_{activation_name}')
                # Mark this position for activation extraction
                should_extract = True
                
                # Skip final output layer if configured to do so
                if self.skip_final_layer:
                    # Check if this is a linear layer with output dimension 1
                    is_final_output = (layer_type == 'linear' and 
                                     layer_config.get('out_features') == 1 and 
                                     len(current_shape) == 1 and 
                                     current_shape[0] == 1)
                    # Also check if this is the last layer with activation
                    is_last_layer = (i == len(self.layers_config) - 1)
                    
                    if is_final_output or (is_last_layer and current_shape[0] == 1):
                        should_extract = False
                
                if should_extract:
                    self.activation_indices.append(len(self.layers) - 1)
            
            # Add batch normalization if specified
            if layer_config.get('batch_norm', False):
                if layer_type == 'linear' or len(current_shape) == 1:
                    self.layers.append(nn.BatchNorm1d(current_shape[0]))
                elif layer_type == 'conv1d' or len(current_shape) == 2:
                    self.layers.append(nn.BatchNorm1d(current_shape[0]))
                elif layer_type == 'conv2d' or len(current_shape) == 3:
                    self.layers.append(nn.BatchNorm2d(current_shape[0]))
                self.layer_types.append('batch_norm')
            
            # Add dropout if specified
            dropout_rate = layer_config.get('dropout', 0.0)
            if dropout_rate > 0:
                if len(current_shape) == 1:
                    self.layers.append(nn.Dropout(dropout_rate))
                elif len(current_shape) == 2:
                    self.layers.append(nn.Dropout1d(dropout_rate))
                elif len(current_shape) == 3:
                    self.layers.append(nn.Dropout2d(dropout_rate))
                self.layer_types.append('dropout')
        
        print(f"Built network with {len(self.layers)} layers")
        print(f"Final output shape: {current_shape}")
        print(f"Activation extraction points: {len(self.activation_indices)} layers")
    
    def _initialize_weights(self):
        """Initialize weights using appropriate methods for each layer type."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor, extract_hidden_activations: bool = False) -> Any:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            extract_hidden_activations: If True, return activations from specified layers
        
        Returns:
            Output tensor, and optionally a list of activation tensors
        """
        hidden_activations = []
        
        # Ensure input has batch dimension
        if x.ndim == len(self.input_shape):
            x = x.unsqueeze(0)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Extract activations after activation functions
            if extract_hidden_activations and i in self.activation_indices:
                if self.extract_from_layers == 'all' or i in self.extract_from_layers:
                    # Clone and detach the activation
                    activation = x.detach().clone()
                    
                    # Flatten conv activations if requested
                    if self.flatten_conv_activations and activation.ndim > 2:
                        # Flatten spatial dimensions but keep batch dimension
                        batch_size = activation.shape[0]
                        activation = activation.view(batch_size, -1)
                    
                    hidden_activations.append(activation)
        
        if extract_hidden_activations:
            return x, hidden_activations
        return x
    
    def extract_layer_outputs(self, data_loader: DataLoader, device: torch.device, 
                             variable_length: bool = False) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Extract layer outputs for the entire dataset.
        
        Args:
            data_loader: DataLoader containing the dataset
            device: Device to run on
            variable_length: If True, return dict with variable-length tensors per layer
        
        Returns:
            If variable_length=False: Tensor of shape (1, num_layers, dataset_size, feature_dim)
            If variable_length=True: Dict mapping layer_idx to tensor of shape (dataset_size, layer_dim)
        """
        self.eval()
        
        print(f"Extracting activations from {len(self.activation_indices)} layers")
        
        # Use CPU for layer extraction to avoid MPS limitations
        extraction_device = torch.device('cpu')
        print(f"Using {extraction_device} for layer extraction (avoiding MPS limitations)")
        
        # Move model to CPU for extraction
        self.to(extraction_device)
        
        # Collect all data
        all_data = []
        for data, _ in data_loader:
            all_data.append(data.to(extraction_device))
        
        # Concatenate all batches
        full_data = torch.cat(all_data, dim=0)
        print(f"Full dataset shape: {full_data.shape}")
        
        # Process entire dataset at once
        with torch.no_grad():
            _, activations = self.forward(full_data, extract_hidden_activations=True)
        
        print(f"Number of activations collected: {len(activations)}")
        for i, act in enumerate(activations):
            print(f"Activation {i} shape: {act.shape}")
        
        if not activations:
            raise RuntimeError("No activations were collected. Check the model architecture.")
        
        if variable_length:
            # Return dictionary with original dimensions preserved
            layer_outputs_dict = {}
            for i, act in enumerate(activations):
                layer_outputs_dict[i] = act.cpu()  # Move to CPU for storage
            return layer_outputs_dict
        else:
            # Original behavior: pad to same dimension
            # Handle different feature dimensions by padding to max dimension
            max_features = max(act.shape[1] for act in activations)
            
            # Pad activations to have same feature dimension
            padded_activations = []
            for act in activations:
                if act.shape[1] < max_features:
                    # Pad with zeros on the right
                    padding = torch.zeros(act.shape[0], max_features - act.shape[1], device=act.device)
                    padded_act = torch.cat([act, padding], dim=1)
                    padded_activations.append(padded_act)
                else:
                    padded_activations.append(act)
            
            # Stack activations to (num_layers, dataset_size, max_feature_dim)
            stacked_activations = torch.stack(padded_activations, dim=0)
            
            # Add batch dimension: (1, num_layers, dataset_size, max_feature_dim)
            output_tensor = stacked_activations.unsqueeze(0)
            
            return output_tensor


class Reshape(nn.Module):
    """Custom reshape layer for nn.Sequential compatibility."""
    
    def __init__(self, target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape
    
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, *self.target_shape)


def train_model(config_path: str):
    """Train a custom architecture model using configuration file."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if custom architecture is enabled
    custom_config = config.get('custom_architecture', {})
    if not custom_config.get('enabled', False):
        raise ValueError("Custom architecture is not enabled in config. Set custom_architecture.enabled: true")
    
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    
    # Device setup
    device_name = training_config['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    print(f"Using device: {device}")
    
    # Create custom model
    model = CustomNet(custom_config).to(device)
    print(f"\nModel architecture:")
    print(model)
    
    # Enable cuDNN benchmarking if using CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Setup optimizer
    lr = training_config['learning_rate']
    opt_config = training_config.get('optimizer', {'type': 'adam'})
    optimizer_type = opt_config.get('type', opt_config.get('name', 'adam')).lower()
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0.0))
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0.01))
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=opt_config.get('weight_decay', 0.0))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    # Regularization parameters
    reg_config = training_config.get('regularization', {})
    l1_lambda = reg_config.get('l1_lambda', 0.0)
    l2_lambda = reg_config.get('l2_lambda', 0.0)
    
    # Loss function (configurable)
    loss_fn_name = training_config.get('loss_fn', 'bce').lower()
    if loss_fn_name == 'bce':
        criterion = nn.BCELoss()
    elif loss_fn_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_fn_name == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_name}")
    
    print(f"Using loss function: {loss_fn_name}")
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Data loading
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
        raise ValueError("Either set data_source or use synthetic data.")
    
    # Move data to device and adjust label format for loss function
    X = X.to(device)
    y = y.to(device)
    
    # Reshape input data to match expected input_shape
    expected_shape = config['custom_architecture']['input_shape']
    if len(expected_shape) == 3 and X.dim() == 2 and X.shape[1] == 784:
        # Reshape flattened MNIST (N, 784) to image format (N, 1, 28, 28)
        if expected_shape == [1, 28, 28] and X.shape[1] == 784:
            X = X.view(-1, 1, 28, 28)
            print(f"Reshaped input from (N, 784) to {X.shape}")
    elif len(expected_shape) == 1 and X.dim() == 2 and X.shape[1] == 784:
        # Keep flattened for 1D input
        pass
    
    # For CrossEntropyLoss, ensure labels are LongTensor with shape (N,) containing class indices
    if loss_fn_name == 'cross_entropy':
        if y.dim() > 1 and y.size(1) == 1:
            y = y.squeeze()  # Remove extra dimension if needed
        y = y.long()  # Convert to LongTensor for CrossEntropyLoss
    elif loss_fn_name == 'bce':
        y = y.float()  # Ensure float for BCELoss
        # For BCELoss, ensure labels have shape (N, 1) to match output
        if y.dim() == 1:
            y = y.unsqueeze(1)  # Add dimension: (N,) -> (N, 1)
    
    # Shuffle data
    # PyTorch 2.5+ has better MPS support, but keep fallback for compatibility
    try:
        perm = torch.randperm(len(X), device=device)
        X = X[perm]
        y = y[perm]
    except Exception as e:
        # Fallback: Use CPU for randperm if MPS has issues
        print(f"Using CPU fallback for shuffling due to: {e}")
        X_cpu = X.cpu()
        y_cpu = y.cpu()
        perm = torch.randperm(len(X_cpu))
        X = X_cpu[perm].to(device)
        y = y_cpu[perm].to(device)
    
    # Split data
    split_ratio = data_config.get('split_ratio', 0.8)
    train_size = int(split_ratio * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config['batch_size'],
        shuffle=True, 
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=training_config['batch_size'],
        shuffle=False, 
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Learning rate scheduler
    scheduler = None
    scheduler_config = training_config.get('scheduler', training_config.get('lr_scheduler', {}))
    if scheduler_config.get('type') == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10)
        )
    
    # Training loop
    print(f"\nStarting training for {training_config['epochs']} epochs...")
    
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
                    # Adjust output shape based on loss function
                    if loss_fn_name == 'cross_entropy':
                        # CrossEntropyLoss expects (N, C) output and (N,) target
                        pass  # Keep output as is
                    else:
                        # BCE/MSE expect matching shapes
                        if output.shape != target.shape:
                            output = output.squeeze(-1)
                    loss = criterion(output, target)
                    
                    # Add regularization
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
                # Adjust output shape based on loss function
                if loss_fn_name == 'cross_entropy':
                    # CrossEntropyLoss expects (N, C) output and (N,) target
                    pass  # Keep output as is
                else:
                    # BCE/MSE expect matching shapes
                    if output.shape != target.shape:
                        output = output.squeeze(-1)
                loss = criterion(output, target)
                
                # Add regularization
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
            
            # Calculate accuracy based on loss function type
            if loss_fn_name == 'cross_entropy':
                # Multi-class classification
                _, predicted = torch.max(output.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()
            else:
                # Binary classification (BCE) or regression (MSE)
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
                # Adjust output shape based on loss function
                if loss_fn_name == 'cross_entropy':
                    # CrossEntropyLoss expects (N, C) output and (N,) target
                    pass  # Keep output as is
                else:
                    # BCE/MSE expect matching shapes
                    if output.shape != target.shape:
                        output = output.squeeze(-1)
                loss = criterion(output, target)
                test_loss_sum += loss.item()
                
                # Calculate accuracy based on loss function type
                if loss_fn_name == 'cross_entropy':
                    # Multi-class classification
                    _, predicted = torch.max(output.data, 1)
                    total_test += target.size(0)
                    correct_test += (predicted == target).sum().item()
                else:
                    # Binary classification (BCE) or regression (MSE)
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
    
    print("\nTraining finished.")
    
    # Save model if enabled and final accuracy meets threshold
    save_model_config = training_config.get('save_model', {})
    if save_model_config.get('enabled', False):
        threshold = save_model_config.get('threshold', 0.0)
        if test_accuracy >= threshold:
            save_dir = Path(save_model_config.get('save_dir', 'results/models'))
            save_dir.mkdir(parents=True, exist_ok=True)
            
            model_filename = f"torch_custom_acc_{test_accuracy:.4f}_epoch_{training_config['epochs']}.pth"
            model_path = save_dir / model_filename
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'custom_config': custom_config,
                'training_config': training_config,
                'final_accuracy': test_accuracy,
                'final_loss': avg_test_loss,
                'epochs_trained': training_config['epochs']
            }, model_path)
            
            print(f"Model saved to: {model_path} (accuracy: {test_accuracy:.4f})")
        else:
            print(f"Model not saved: accuracy {test_accuracy:.4f} below threshold {threshold:.4f}")
    
    # Extract layer outputs if enabled
    layer_extraction_config = config.get('layer_extraction', {})
    if layer_extraction_config.get('enabled', False):
        print("\nExtracting layer outputs...")
        model.to(device)
        
        # Combine train and test datasets
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        full_loader = DataLoader(
            full_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Check if variable length output is requested
        variable_length = layer_extraction_config.get('variable_length_output', False)
        layer_outputs = model.extract_layer_outputs(full_loader, device, variable_length=variable_length)
        
        if variable_length:
            print(f"Extracted variable-length layer outputs: {len(layer_outputs)} layers")
            for idx, tensor in layer_outputs.items():
                print(f"  Layer {idx}: {tensor.shape}")
        else:
            print(f"Shape of extracted layer outputs: {layer_outputs.shape}")
        
        # Save layer outputs
        output_dir = Path(layer_extraction_config.get('output_dir', 'results/layer_outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if variable_length:
            # Save with different filename to distinguish format
            output_file = output_dir / 'torch_custom_layer_outputs_varlen.pt'
        else:
            output_file = output_dir / 'torch_custom_layer_outputs.pt'
        
        torch.save({
            'layer_outputs': layer_outputs.cpu() if not variable_length else layer_outputs,
            'config': config,
            'variable_length': variable_length
        }, output_file)
        print(f"Layer outputs saved to: {output_file}")
    
    # Extract train/test activations separately if enabled
    if layer_extraction_config.get('train_test_layer_extraction', False):
        print("\nExtracting train and test layer outputs separately...")
        
        train_test_output_dir = Path(layer_extraction_config.get('train_test_output_dir', 'results/train_test_layer_outputs'))
        train_test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract train dataset activations
        print("Extracting train dataset activations...")
        train_layer_outputs = model.extract_layer_outputs(train_loader, device, variable_length=variable_length)
        
        if variable_length:
            train_output_file = train_test_output_dir / 'torch_custom_train_layer_outputs_varlen.pt'
        else:
            train_output_file = train_test_output_dir / 'torch_custom_train_layer_outputs.pt'
            
        torch.save({
            'layer_outputs': train_layer_outputs.cpu() if not variable_length else train_layer_outputs,
            'dataset_type': 'train',
            'dataset_size': len(train_dataset),
            'config': config,
            'variable_length': variable_length
        }, train_output_file)
        print(f"Train layer outputs saved to: {train_output_file}")
        
        # Extract test dataset activations
        print("Extracting test dataset activations...")
        test_layer_outputs = model.extract_layer_outputs(test_loader, device, variable_length=variable_length)
        
        if variable_length:
            test_output_file = train_test_output_dir / 'torch_custom_test_layer_outputs_varlen.pt'
        else:
            test_output_file = train_test_output_dir / 'torch_custom_test_layer_outputs.pt'
            
        torch.save({
            'layer_outputs': test_layer_outputs.cpu() if not variable_length else test_layer_outputs,
            'dataset_type': 'test',
            'dataset_size': len(test_dataset),
            'config': config,
            'variable_length': variable_length
        }, test_output_file)
        print(f"Test layer outputs saved to: {test_output_file}")
        
        print(f"Train/test layer extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom architecture model using YAML configuration.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    train_model(args.config_path)