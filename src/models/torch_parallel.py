import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import multiprocessing as mp
import numpy as np
from pathlib import Path
import sys
import os
import time
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import tempfile

# --- Dynamic Project Root Addition ---
try:
    current_file_path = Path(__file__).resolve()
    project_root_found = False
    _temp_path = current_file_path
    for _ in range(5): 
        if (_temp_path.parent / 'src').is_dir():
            project_root = _temp_path.parent
            if str(project_root) not in sys.path:
                 sys.path.append(str(project_root))
            print(f"Project root identified and added to sys.path: {project_root}")
            project_root_found = True
            break
        _temp_path = _temp_path.parent
    if not project_root_found:
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        print(f"Falling back to project root (original logic): {project_root}")

    from src.models.torch_mlp import MLP, generate_torus_data, load_data_from_file
    print("Successfully imported MLP, generate_torus_data, and load_data_from_file.")
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"Searched for 'src' directory upwards from {current_file_path if '__file__' in locals() else 'current script directory'}.")
    print("Please ensure your project structure is correct or adjust PYTHONPATH.")
    if 'src.models.torch_mlp' not in sys.modules:
         raise ImportError(
             "Could not import MLP and generate_torus_data from src.models.torch_mlp. "
             "Check 'src' directory in Python path and 'models/torch_mlp.py'."
         )
except NameError: 
    print("Warning: __file__ not defined. Relative imports might fail. Ensure 'src' is in PYTHONPATH.")
    from src.models.torch_mlp import MLP, generate_torus_data, load_data_from_file


def train_single_network(args):
    """
    Train a single network in a separate process.
    
    Args:
        args: Tuple containing (network_id, config, data_info, random_seed)
    
    Returns:
        Tuple containing (network_id, final_loss, final_accuracy, layer_outputs, training_time)
    """
    network_id, config, data_info, random_seed = args
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed + network_id)
    np.random.seed(random_seed + network_id)
    
    # Extract configuration
    model_config = config['model']
    training_config = config['training']
    
    # Always use CPU for parallel processing
    device = torch.device('cpu')
    
    # Create model
    model = MLP(
        input_dim=model_config['input_dim'],
        num_hidden_layers=model_config['num_hidden_layers'],
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        activation_fn_name=model_config.get('activation_fn_name', 'relu'),
        dropout_rate=model_config.get('dropout_rate', 0.0),
        use_batch_norm=model_config.get('use_batch_norm', False)
    ).to(device)
    
    # Load data
    X_train = torch.from_numpy(data_info['X_train']).float()
    y_train = torch.from_numpy(data_info['y_train']).float()
    X_test = torch.from_numpy(data_info['X_test']).float()
    y_test = torch.from_numpy(data_info['y_test']).float()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0  # Disable nested multiprocessing
    )
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Setup optimizer
    lr = training_config['learning_rate']
    opt_config = training_config.get('optimizer', {'name': 'adam'})
    optimizer_name = opt_config.get('name', 'adam').lower()
    weight_decay = opt_config.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay if weight_decay > 0 else 0.01)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Setup learning rate scheduler
    scheduler = None
    scheduler_config = training_config.get('lr_scheduler', {})
    if scheduler_config.get('type') == 'reduce_on_plateau':
        scheduler_kwargs = {
            'factor': scheduler_config.get('factor', 0.1),
            'patience': scheduler_config.get('patience', 10),
            'min_lr': scheduler_config.get('min_lr', 0)
        }
        # Only add verbose if the PyTorch version supports it
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                verbose=scheduler_config.get('verbose', False),
                **scheduler_kwargs
            )
        except TypeError:
            # Fallback for older PyTorch versions without verbose parameter
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **scheduler_kwargs
            )
    elif scheduler_config.get('type') == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    
    # Setup loss function
    loss_fn_name = training_config.get('loss_fn', 'bce').lower()
    if loss_fn_name == 'bce':
        criterion = nn.BCELoss()
    elif loss_fn_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_fn_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_name}")
    
    # Setup gradient clipping
    gradient_clipping_config = training_config.get('gradient_clipping', {})
    use_gradient_clipping = gradient_clipping_config.get('enabled', False)
    max_norm = gradient_clipping_config.get('max_norm', 1.0)
    
    # Setup early stopping
    early_stopping_config = training_config.get('early_stopping', {})
    use_early_stopping = early_stopping_config.get('enabled', False)
    early_stopping_patience = early_stopping_config.get('patience', 10)
    min_delta = early_stopping_config.get('min_delta', 0.0)
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    start_time = time.time()
    final_loss = 0.0
    final_accuracy = 0.0
    
    for epoch in range(training_config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Apply gradient clipping if enabled
            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            train_loss += loss.item()
            if loss_fn_name == 'bce':
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                if loss_fn_name == 'bce':
                    predicted = (outputs > 0.5).float()
                    test_total += batch_y.size(0)
                    test_correct += (predicted == batch_y).sum().item()
        
        avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        test_accuracy = test_correct / test_total if test_total > 0 else 0.0
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_test_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        if use_early_stopping:
            if avg_test_loss < best_loss - min_delta:
                best_loss = avg_test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break
        
        final_loss = avg_test_loss
        final_accuracy = test_accuracy
    
    training_time = time.time() - start_time
    
    # Save model if it meets the threshold
    save_threshold = training_config.get('save_model_threshold', 0.0)
    model_saved = False
    if final_accuracy >= save_threshold:
        # Save model to temporary location (will be moved by main process if needed)
        temp_dir = Path('/tmp/torch_parallel_models')
        temp_dir.mkdir(exist_ok=True)
        model_path = temp_dir / f'network_{network_id}_acc_{final_accuracy:.4f}.pth'
        torch.save(model.state_dict(), model_path)
        model_saved = True
    
    # Extract layer outputs if needed
    layer_outputs = None
    layer_extraction_config = config.get('layer_extraction', {})
    if layer_extraction_config.get('enabled', False):
        # Combine train and test data for layer extraction
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        full_loader = DataLoader(full_dataset, batch_size=training_config['batch_size'], shuffle=False, num_workers=0)
        
        layer_outputs_tensor = model.extract_layer_outputs(full_loader, device)
        # Convert to numpy for efficient serialization
        layer_outputs = layer_outputs_tensor.cpu().numpy()
    
    return network_id, final_loss, final_accuracy, layer_outputs, training_time, model_saved


class ParallelTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.data_config = self.config['data']
        
        self.num_networks = self.training_config.get('num_networks', 1)
        self.max_workers = self.training_config.get('max_parallel_workers')
        if self.max_workers is None:
            self.max_workers = min(mp.cpu_count(), self.num_networks)
        
        print(f"Initializing ParallelTrainer with {self.num_networks} networks")
        print(f"Using {self.max_workers} parallel workers")
    
    def prepare_data(self):
        """Prepare and split data for training."""
        print("Preparing data...")
        
        # Data generation or loading
        data_source = self.data_config.get('data_source')
        if data_source is not None:
            print(f"Loading data from: {data_source}")
            X, y = load_data_from_file(data_source)
        elif self.data_config['type'] == 'synthetic' and self.data_config.get('synthetic_type') == 'torus':
            num_samples = self.data_config.get('generation', {}).get('n', 1000)
            big_radius = self.data_config.get('generation', {}).get('big_radius', 3)
            small_radius = self.data_config.get('generation', {}).get('small_radius', 1)
            X, y = generate_torus_data(num_samples, big_radius, small_radius)
        else:
            raise ValueError(f"Unsupported data configuration. Either set data_source or use synthetic data.")
        
        # Shuffle data if enabled (following torch_mlp.py pattern)
        if self.data_config.get('shuffle_data', True):
            seed = self.data_config.get('random_seed_data', 42)
            torch.manual_seed(seed)
            perm = torch.randperm(len(X))
            X = X[perm]
            y = y[perm]
        
        # Split data
        split_ratio = self.data_config.get('split_ratio', 0.8)
        train_size = int(split_ratio * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to numpy for efficient sharing between processes
        self.data_info = {
            'X_train': X_train.numpy(),
            'y_train': y_train.numpy(),
            'X_test': X_test.numpy(),
            'y_test': y_test.numpy()
        }
        
        print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        
    def train(self):
        """Train multiple networks in parallel."""
        self.prepare_data()
        
        print(f"\nStarting parallel training of {self.num_networks} networks...")
        print(f"Epochs: {self.training_config['epochs']}, Batch size: {self.training_config['batch_size']}")
        
        # Prepare arguments for each network
        base_seed = self.training_config.get('seed', 42)
        train_args = [
            (i, self.config, self.data_info, base_seed)
            for i in range(self.num_networks)
        ]
        
        # Train networks in parallel
        start_time = time.time()
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all training tasks
            future_to_id = {
                executor.submit(train_single_network, args): args[0] 
                for args in train_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_id):
                network_id = future_to_id[future]
                try:
                    result = future.result()
                    results.append(result)
                    _, final_loss, final_accuracy, _, training_time, model_saved = result
                    saved_str = " [SAVED]" if model_saved else ""
                    print(f"Network {network_id:3d} completed - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}, Time: {training_time:.2f}s{saved_str}")
                except Exception as exc:
                    print(f"Network {network_id} generated an exception: {exc}")
        
        total_time = time.time() - start_time
        
        # Sort results by network ID
        results.sort(key=lambda x: x[0])
        
        # Compute statistics
        losses = [r[1] for r in results]
        accuracies = [r[2] for r in results]
        training_times = [r[4] for r in results]
        models_saved = sum(1 for r in results if r[5])
        
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Average loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"Average accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Average training time per network: {np.mean(training_times):.2f}s")
        print(f"Parallel efficiency: {sum(training_times)/total_time:.1f}x")
        print(f"Models saved: {models_saved}/{self.num_networks}")
        
        # Save layer outputs if enabled
        layer_extraction_config = self.config.get('layer_extraction', {})
        if layer_extraction_config.get('enabled', False):
            self.save_layer_outputs(results)
        
        return results
    
    def save_layer_outputs(self, results):
        """Save extracted layer outputs from all networks."""
        print("\nSaving layer outputs...")
        
        # Extract layer outputs from results
        layer_outputs_list = []
        for network_id, _, _, layer_outputs, _, _ in results:
            if layer_outputs is not None:
                layer_outputs_list.append(layer_outputs)
        
        if not layer_outputs_list:
            print("No layer outputs to save.")
            return
        
        # Stack all layer outputs: (num_networks, num_layers, num_samples, hidden_dim)
        all_layer_outputs = np.stack(layer_outputs_list, axis=0)
        
        # Create output directory
        output_dir = Path(self.config.get('layer_extraction', {}).get('output_dir', 'results/layer_outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as PyTorch tensor
        output_file = output_dir / 'torch_parallel_cpu_layer_outputs.pt'
        torch.save({
            'layer_outputs': torch.from_numpy(all_layer_outputs),
            'config': self.config,
            'num_networks': self.num_networks,
            'statistics': {
                'losses': [r[1] for r in results],
                'accuracies': [r[2] for r in results],
                'training_times': [r[4] for r in results],
                'models_saved': [r[5] for r in results]
            }
        }, output_file)
        
        print(f"Layer outputs saved to: {output_file}")
        print(f"Shape: {all_layer_outputs.shape}")


if __name__ == "__main__":
    # Ensure proper multiprocessing start method
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Train multiple MLP models in parallel using multiprocessing.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    trainer = ParallelTrainer(args.config_path)
    trainer.train()