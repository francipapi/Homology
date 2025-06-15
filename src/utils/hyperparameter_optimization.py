import os
import sys
import yaml
import torch
import optuna
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import vectorization utilities
try:
    from torch.func import make_functional_with_buffers, vmap
    VMAP_AVAILABLE = True
except ImportError:
    try:
        from functorch import make_functional_with_buffers, vmap
        VMAP_AVAILABLE = True
    except ImportError:
        VMAP_AVAILABLE = False
        print("⚠️  Warning: vmap not available. GPU vectorized optimization disabled.")

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.torch_mlp import MLP, generate_torus_data, load_data_from_file
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


def train_single_trial(trial_config: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Train a single model with given hyperparameters and return validation metrics.
    
    Args:
        trial_config: Dictionary containing all training configuration
        
    Returns:
        Tuple of (validation_loss, validation_accuracy, training_time)
    """
    import time
    start_time = time.time()
    
    # Extract configurations
    model_config = trial_config['model']
    training_config = trial_config['training']
    data_config = trial_config['data']
    
    # Set random seed for reproducibility
    torch.manual_seed(trial_config.get('seed', 42))
    np.random.seed(trial_config.get('seed', 42))
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # Create optimizer
    lr = training_config['learning_rate']
    optimizer_config = training_config.get('optimizer', {})
    optimizer_name = optimizer_config.get('name', 'adam').lower()
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Use cached data if available, otherwise load/generate
    if 'cached_data' in trial_config:
        X, y = trial_config['cached_data']
        X = X.to(device)
        y = y.to(device)
    else:
        # Load or generate data
        data_source = data_config.get('data_source')
        if data_source is not None:
            X, y = load_data_from_file(data_source)
        elif data_config['type'] == 'synthetic':
            num_samples = data_config.get('generation', {}).get('n', 1000)
            big_radius = data_config.get('generation', {}).get('big_radius', 3)
            small_radius = data_config.get('generation', {}).get('small_radius', 1)
            X, y = generate_torus_data(num_samples, big_radius, small_radius)
        else:
            raise ValueError("Invalid data configuration")
        
        # Move data to device
        X = X.to(device)
        y = y.to(device)
    
    # Shuffle data (with MPS compatibility)
    if device.type == 'mps':
        # MPS doesn't support randperm yet, use CPU then move to device
        perm = torch.randperm(len(X), device='cpu')
        X = X.cpu()[perm].to(device)
        y = y.cpu()[perm].to(device)
    else:
        perm = torch.randperm(len(X), device=device)
        X = X[perm]
        y = y[perm]
    
    # Split data
    split_ratio = data_config.get('split_ratio', 0.8)
    train_size = int(split_ratio * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = training_config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get regularization parameters
    reg_config = training_config.get('regularization', {})
    l1_lambda = reg_config.get('l1_lambda', 0.0)
    l2_lambda = reg_config.get('l2_lambda', 0.0)
    
    # Training loop
    num_epochs = training_config.get('epochs', 100)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_sum = 0
        correct_train = 0
        total_train = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
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
            
            # Gradient clipping
            if training_config.get('gradient_clipping', {}).get('enabled', False):
                max_norm = training_config['gradient_clipping'].get('max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            train_loss_sum += loss.item()
            predicted = (output > 0.5).float()
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss_sum = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss_sum += loss.item()
                
                predicted = (output > 0.5).float()
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        val_accuracy = correct_val / total_val
        
        # Update best metrics
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
    
    training_time = time.time() - start_time
    
    return best_val_loss, best_val_acc, training_time


def parallel_trial_worker(args):
    """
    Worker function for parallel trial execution.
    
    Args:
        args: Tuple of (study_storage, study_name, base_config, cached_data)
    
    Returns:
        Tuple of (trial_number, trial_value, trial_params, trial_state)
    """
    study_storage, study_name, base_config, cached_data = args
    
    # Load study in worker process
    study = optuna.load_study(
        study_name=study_name,
        storage=study_storage
    )
    
    # Create optimizer instance for this worker
    class WorkerOptimizer:
        def __init__(self, base_config, cached_data):
            self.base_config = base_config
            self._cached_data = cached_data
            
        def objective(self, trial: optuna.Trial) -> float:
            # Same objective function as the main class but standalone
            trial_config = deepcopy(self.base_config)
            
            # Sample hyperparameters
            trial_config['training']['learning_rate'] = trial.suggest_float(
                'learning_rate', 1e-5, 1e-1, log=True
            )
            trial_config['training']['batch_size'] = trial.suggest_categorical(
                'batch_size', [16, 32, 64, 128, 256]
            )
            trial_config['model']['dropout_rate'] = trial.suggest_float(
                'dropout_rate', 0.0, 0.5
            )
            trial_config['training']['regularization'] = {
                'l1_lambda': trial.suggest_float('l1_lambda', 1e-6, 1e-1, log=True),
                'l2_lambda': trial.suggest_float('l2_lambda', 1e-6, 1e-1, log=True)
            }
            optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
            trial_config['training']['optimizer']['name'] = optimizer_name
            trial_config['training']['optimizer']['weight_decay'] = trial.suggest_float(
                'weight_decay', 1e-6, 1e-2, log=True
            )
            if optimizer_name == 'sgd':
                trial_config['training']['optimizer']['momentum'] = trial.suggest_float(
                    'momentum', 0.8, 0.99
                )
            use_grad_clip = trial.suggest_categorical('use_gradient_clipping', [True, False])
            if use_grad_clip:
                trial_config['training']['gradient_clipping'] = {
                    'enabled': True,
                    'max_norm': trial.suggest_float('grad_clip_max_norm', 0.1, 5.0)
                }
            else:
                trial_config['training']['gradient_clipping'] = {'enabled': False}
            
            trial_config['training']['epochs'] = trial.suggest_int('epochs', 30, 100)
            trial_config['seed'] = trial.number
            trial_config['cached_data'] = self._cached_data
            
            try:
                # Use simplified training without pruning for parallel execution
                val_loss, val_acc, training_time = train_single_trial(trial_config)
                trial.set_user_attr('validation_loss', val_loss)
                trial.set_user_attr('training_time', training_time)
                return val_acc
            except Exception as e:
                print(f"Worker trial {trial.number} failed: {e}")
                return 0.0
    
    # Run one trial
    optimizer = WorkerOptimizer(base_config, cached_data)
    try:
        study.optimize(optimizer.objective, n_trials=1)
        # Return the latest trial info
        latest_trial = study.trials[-1]
        return (latest_trial.number, latest_trial.value, latest_trial.params, latest_trial.state)
    except Exception as e:
        print(f"Worker process failed: {e}")
        return None


class VectorizedOptimizer:
    """
    GPU-based vectorized hyperparameter optimization.
    Trains multiple networks with different hyperparameters simultaneously using vmap.
    """
    
    def __init__(self, base_config: Dict[str, Any], cached_data: Tuple[torch.Tensor, torch.Tensor], 
                 device: torch.device, opt_config: Dict[str, Any]):
        """
        Initialize vectorized optimizer.
        
        Args:
            base_config: Base training configuration
            cached_data: Pre-loaded dataset (X, y)
            device: GPU device to use
            opt_config: Optimization configuration
        """
        if not VMAP_AVAILABLE:
            raise RuntimeError("vmap not available. Cannot use vectorized optimization.")
        
        self.base_config = base_config
        self.cached_data = cached_data
        self.device = device
        self.opt_config = opt_config
        
        # GPU-specific settings
        gpu_config = opt_config.get('gpu', {})
        self.batch_size = gpu_config.get('vectorized_batch_size', 8)
        self.use_mixed_precision = gpu_config.get('use_mixed_precision', True) and device.type == 'cuda'
        
        print(f"🔥 Initializing vectorized optimizer:")
        print(f"   • Device: {device}")
        print(f"   • Batch size: {self.batch_size} networks")
        print(f"   • Mixed precision: {self.use_mixed_precision} {'(CUDA only)' if device.type != 'cuda' else ''}")
        
        # Move data to device
        self.X, self.y = cached_data
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        
        # Split data
        data_config = base_config['data']
        split_ratio = data_config.get('split_ratio', 0.8)
        train_size = int(split_ratio * len(self.X))
        
        # Shuffle data (with MPS compatibility)
        if device.type == 'mps':
            # MPS doesn't support randperm yet, use CPU then move to device
            perm = torch.randperm(len(self.X), device='cpu')
            self.X = self.X.cpu()[perm].to(device)
            self.y = self.y.cpu()[perm].to(device)
        else:
            perm = torch.randperm(len(self.X), device=device)
            self.X = self.X[perm]
            self.y = self.y[perm]
        
        self.X_train, self.X_val = self.X[:train_size], self.X[train_size:]
        self.y_train, self.y_val = self.y[:train_size], self.y[train_size:]
        
        print(f"   • Training samples: {len(self.X_train)}")
        print(f"   • Validation samples: {len(self.X_val)}")
    
    def functional_forward_pass_template(self, functional_model, params, buffers, x):
        """Template for vmapped forward pass."""
        return functional_model(params, buffers, x)
    
    def create_vectorized_models(self, param_sets: list) -> Tuple:
        """
        Create vectorized models for a batch of hyperparameter sets.
        
        Args:
            param_sets: List of parameter dictionaries
            
        Returns:
            Tuple of (vmapped_forward, stacked_params, stacked_buffers, optimizers)
        """
        model_config = self.base_config['model']
        
        # Create individual models with different hyperparameters
        models = []
        optimizers = []
        
        for params in param_sets:
            # Create model with specific hyperparameters
            model = MLP(
                input_dim=model_config['input_dim'],
                num_hidden_layers=model_config['num_hidden_layers'],
                hidden_dim=model_config['hidden_dim'],
                output_dim=model_config['output_dim'],
                activation_fn_name=model_config.get('activation_fn_name', 'relu'),
                dropout_rate=params.get('dropout_rate', 0.0),
                use_batch_norm=model_config.get('use_batch_norm', False)
            ).to(self.device)
            
            models.append(model)
            
            # Create optimizer for this model
            lr = params['learning_rate']
            optimizer_name = params['optimizer']
            weight_decay = params['weight_decay']
            
            if optimizer_name == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name == 'sgd':
                momentum = params.get('momentum', 0.9)
                optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
            else:
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            optimizers.append(optimizer)
        
        # Convert models to functional form
        all_params = []
        all_buffers = []
        functional_model = None
        
        for i, model in enumerate(models):
            if i == 0:
                functional_model, params_template, buffers_template = make_functional_with_buffers(
                    model, disable_autograd_tracking=True
                )
                current_params = params_template
                current_buffers = buffers_template
            else:
                _, current_params, current_buffers = make_functional_with_buffers(
                    model, disable_autograd_tracking=True
                )
            
            all_params.append(list(current_params))
            all_buffers.append(list(current_buffers))
        
        # Stack parameters and buffers
        stacked_params = [
            torch.stack([model_params[i] for model_params in all_params]).requires_grad_(True)
            for i in range(len(params_template))
        ]
        
        if len(buffers_template) > 0:
            stacked_buffers = [
                torch.stack([model_buffers[i] for model_buffers in all_buffers])
                for i in range(len(buffers_template))
            ]
        else:
            stacked_buffers = []
        
        # Create vmapped forward function
        vmapped_forward = vmap(
            lambda p_tuple, b_tuple, x: self.functional_forward_pass_template(
                functional_model, p_tuple, b_tuple, x
            ),
            in_dims=(0, 0, None),
            out_dims=0,
            randomness='different'
        )
        
        return vmapped_forward, stacked_params, stacked_buffers, optimizers
    
    def train_batch(self, param_sets: list, n_epochs: int = 50) -> list:
        """
        Train a batch of networks with different hyperparameters.
        
        Args:
            param_sets: List of hyperparameter dictionaries
            n_epochs: Number of epochs to train
            
        Returns:
            List of validation accuracies
        """
        try:
            # Create vectorized models
            vmapped_forward, stacked_params, stacked_buffers, _ = self.create_vectorized_models(param_sets)
            
            # Create single optimizer for all stacked parameters
            base_lr = sum(params['learning_rate'] for params in param_sets) / len(param_sets)
            optimizer = optim.Adam(stacked_params, lr=base_lr)
            
            # Loss function
            criterion = nn.BCELoss(reduction='none')
            
            # Training parameters
            batch_size = self.base_config['training']['batch_size']
            
            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            val_dataset = torch.utils.data.TensorDataset(self.X_val, self.y_val)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
            
            # Training loop
            best_val_accs = [0.0] * len(param_sets)
            
            for epoch in range(n_epochs):
                # Training phase
                total_loss = 0
                num_batches = 0
                
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass through all networks simultaneously
                    # predictions shape: (num_networks, batch_size, output_dim)
                    if self.use_mixed_precision and self.device.type == 'cuda':
                        # Mixed precision only supported on CUDA
                        with torch.cuda.amp.autocast():
                            predictions = vmapped_forward(stacked_params, stacked_buffers, batch_x)
                    else:
                        predictions = vmapped_forward(stacked_params, stacked_buffers, batch_x)
                    
                    # Prepare targets
                    if batch_y.ndim == 1:
                        batch_y = batch_y.unsqueeze(-1)
                    
                    # Expand targets for all networks
                    batch_y_expanded = batch_y.unsqueeze(0).expand_as(predictions)
                    
                    # Compute loss for all networks
                    if predictions.shape[-1] == 1:
                        predictions = predictions.squeeze(-1)
                        batch_y_expanded = batch_y_expanded.squeeze(-1)
                    
                    losses = criterion(predictions, batch_y_expanded)
                    mean_losses = losses.mean(dim=-1) if losses.ndim > 1 else losses
                    total_loss = mean_losses.mean()
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    num_batches += 1
                
                # Validation phase every 10 epochs
                if epoch % 10 == 0 or epoch == n_epochs - 1:
                    with torch.no_grad():
                        val_correct = torch.zeros(len(param_sets), device=self.device)
                        val_total = 0
                        
                        for batch_x, batch_y in val_loader:
                            predictions = vmapped_forward(stacked_params, stacked_buffers, batch_x)
                            
                            if batch_y.ndim == 1:
                                batch_y = batch_y.unsqueeze(-1)
                            
                            batch_y_expanded = batch_y.unsqueeze(0).expand_as(predictions)
                            
                            if predictions.shape[-1] == 1:
                                predictions = predictions.squeeze(-1)
                                batch_y_expanded = batch_y_expanded.squeeze(-1)
                            
                            # Binary classification accuracy
                            pred_binary = (predictions > 0.5).float()
                            correct = (pred_binary == batch_y_expanded).float()
                            val_correct += correct.sum(dim=-1)  # Sum over batch dimension
                            val_total += batch_y.size(0)
                        
                        val_accs = (val_correct / val_total).cpu().tolist()
                        
                        # Update best accuracies
                        for i, acc in enumerate(val_accs):
                            if acc > best_val_accs[i]:
                                best_val_accs[i] = acc
            
            return best_val_accs
            
        except Exception as e:
            print(f"❌ Batch training failed: {e}")
            return [0.0] * len(param_sets)


class HyperparameterOptimizer:
    """
    Bayesian optimization for hyperparameter tuning using Optuna.
    """
    
    def __init__(self, base_config_path: str, optimization_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the optimizer.
        
        Args:
            base_config_path: Path to the base training configuration YAML file
            optimization_config: Optional optimization configuration dictionary
        """
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Default optimization configuration
        self.opt_config = optimization_config or {
            'n_trials': 100,
            'device': 'auto',
            'cpu': {'n_jobs': 4},
            'gpu': {
                'vectorized_batch_size': 8,
                'use_mixed_precision': True,
                'max_memory_fraction': 0.8
            },
            'study_name': 'torch_mlp_optimization',
            'storage': 'sqlite:///optuna_study.db',
            'pruner': {
                'type': 'median',
                'n_startup_trials': 5,
                'n_warmup_steps': 10
            }
        }
        
        # Device detection and setup
        self.device = self._setup_device()
        
        # Create pruner
        pruner_config = self.opt_config.get('pruner', {})
        if pruner_config.get('type') == 'median':
            self.pruner = optuna.pruners.MedianPruner(
                n_startup_trials=pruner_config.get('n_startup_trials', 5),
                n_warmup_steps=pruner_config.get('n_warmup_steps', 10)
            )
        else:
            self.pruner = None
        
        # Create or load study with proper error handling
        try:
            self.study = optuna.create_study(
                study_name=self.opt_config['study_name'],
                storage=self.opt_config.get('storage'),
                direction='maximize',  # Maximize validation accuracy
                pruner=self.pruner,
                load_if_exists=True
            )
        except Exception as e:
            print(f"⚠️  Database issue detected: {e}")
            print("🔧 Creating fresh database...")
            # Remove corrupted database and create new one
            import os
            db_path = self.opt_config.get('storage', '').replace('sqlite:///', '')
            if os.path.exists(db_path):
                os.remove(db_path)
            
            self.study = optuna.create_study(
                study_name=self.opt_config['study_name'],
                storage=self.opt_config.get('storage'),
                direction='maximize',
                pruner=self.pruner,
                load_if_exists=False
            )
        
        # Cache data to avoid regenerating each trial
        self._cached_data = None
        self._cache_data()
    
    def _setup_device(self):
        """Setup and validate the compute device for optimization."""
        device_config = self.opt_config.get('device', 'auto')
        
        if device_config == 'auto':
            # Auto-detect best available device
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"🚀 Auto-detected CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                print(f"🚀 Auto-detected MPS device (Apple Silicon)")
            else:
                device = torch.device('cpu')
                print(f"🚀 Using CPU device")
        elif device_config == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                print("⚠️  CUDA not available, falling back to CPU")
                device = torch.device('cpu')
        elif device_config == 'mps':
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                print(f"🚀 Using MPS device (Apple Silicon)")
            else:
                print("⚠️  MPS not available, falling back to CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            print(f"🚀 Using CPU device")
        
        # Set optimization strategy based on device
        if device.type == 'cpu':
            self.optimization_strategy = 'multiprocessing'
            n_jobs = self.opt_config.get('cpu', {}).get('n_jobs', 4)
            print(f"📊 Strategy: Multiprocessing with {n_jobs} workers")
        else:
            self.optimization_strategy = 'vectorized'
            batch_size = self.opt_config.get('gpu', {}).get('vectorized_batch_size', 8)
            print(f"📊 Strategy: Vectorized training with batch size {batch_size}")
        
        return device
    
    def _cache_data(self):
        """Cache the dataset to avoid regenerating it for each trial."""
        print("📊 Loading/generating dataset (one-time operation)...")
        data_config = self.base_config['data']
        
        # Load or generate data
        data_source = data_config.get('data_source')
        if data_source is not None:
            X, y = load_data_from_file(data_source)
        elif data_config['type'] == 'synthetic':
            num_samples = data_config.get('generation', {}).get('n', 1000)
            big_radius = data_config.get('generation', {}).get('big_radius', 3)
            small_radius = data_config.get('generation', {}).get('small_radius', 1)
            X, y = generate_torus_data(num_samples, big_radius, small_radius)
        else:
            raise ValueError("Invalid data configuration")
        
        # Store as CPU tensors
        self._cached_data = (X.cpu(), y.cpu())
        print(f"✅ Dataset cached: {X.shape[0]} samples")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy (to maximize)
        """
        # Create trial configuration by copying base config
        trial_config = deepcopy(self.base_config)
        
        # Sample hyperparameters
        # Learning rate
        trial_config['training']['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-5, 1e-1, log=True
        )
        
        # Batch size
        trial_config['training']['batch_size'] = trial.suggest_categorical(
            'batch_size', [16, 32, 64, 128, 256]
        )
        
        # Dropout rate
        trial_config['model']['dropout_rate'] = trial.suggest_float(
            'dropout_rate', 0.0, 0.5
        )
        
        # Regularization
        trial_config['training']['regularization'] = {
            'l1_lambda': trial.suggest_float('l1_lambda', 1e-6, 1e-1, log=True),
            'l2_lambda': trial.suggest_float('l2_lambda', 1e-6, 1e-1, log=True)
        }
        
        # Optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        trial_config['training']['optimizer']['name'] = optimizer_name
        
        # Weight decay
        trial_config['training']['optimizer']['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-6, 1e-2, log=True
        )
        
        # SGD specific parameters
        if optimizer_name == 'sgd':
            trial_config['training']['optimizer']['momentum'] = trial.suggest_float(
                'momentum', 0.8, 0.99
            )
        
        # Gradient clipping
        use_grad_clip = trial.suggest_categorical('use_gradient_clipping', [True, False])
        if use_grad_clip:
            trial_config['training']['gradient_clipping'] = {
                'enabled': True,
                'max_norm': trial.suggest_float('grad_clip_max_norm', 0.1, 5.0)
            }
        else:
            trial_config['training']['gradient_clipping'] = {'enabled': False}
        
        # Number of epochs - reduced for optimization speed
        trial_config['training']['epochs'] = trial.suggest_int('epochs', 30, 100)
        
        # Set random seed for reproducibility
        trial_config['seed'] = trial.number
        
        # Pass cached data to avoid regeneration
        trial_config['cached_data'] = self._cached_data
        
        try:
            # Train model with sampled hyperparameters and pruning support
            val_loss, val_acc, training_time = self._train_with_pruning(trial, trial_config)
            
            # Log metrics
            trial.set_user_attr('validation_loss', val_loss)
            trial.set_user_attr('training_time', training_time)
            
            return val_acc
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            return 0.0  # Return worst possible value
    
    def _train_with_pruning(self, trial: optuna.Trial, trial_config: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Train model with pruning support for early stopping of bad trials.
        """
        import time
        start_time = time.time()
        
        # Extract configurations
        model_config = trial_config['model']
        training_config = trial_config['training']
        
        # Set random seed
        torch.manual_seed(trial_config.get('seed', 42))
        np.random.seed(trial_config.get('seed', 42))
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        # Create optimizer
        lr = training_config['learning_rate']
        optimizer_config = training_config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adam').lower()
        weight_decay = optimizer_config.get('weight_decay', 0.0)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Use cached data
        X, y = trial_config['cached_data']
        X = X.to(device)
        y = y.to(device)
        
        # Shuffle data
        perm = torch.randperm(len(X), device=device)
        X = X[perm]
        y = y[perm]
        
        # Split data
        split_ratio = trial_config['data'].get('split_ratio', 0.8)
        train_size = int(split_ratio * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        batch_size = training_config['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Get regularization parameters
        reg_config = training_config.get('regularization', {})
        l1_lambda = reg_config.get('l1_lambda', 0.0)
        l2_lambda = reg_config.get('l2_lambda', 0.0)
        
        # Training loop with pruning
        num_epochs = training_config.get('epochs', 100)
        best_val_loss = float('inf')
        best_val_acc = 0.0
        
        # Early stopping parameters
        patience = 10
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss_sum = 0
            
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
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
                
                # Gradient clipping
                if training_config.get('gradient_clipping', {}).get('enabled', False):
                    max_norm = training_config['gradient_clipping'].get('max_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                optimizer.step()
                train_loss_sum += loss.item()
            
            # Validation phase
            model.eval()
            val_loss_sum = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss_sum += loss.item()
                    
                    predicted = (output > 0.5).float()
                    total_val += target.size(0)
                    correct_val += (predicted == target).sum().item()
            
            avg_val_loss = val_loss_sum / len(val_loader)
            val_accuracy = correct_val / total_val
            
            # Update best metrics
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            
            # Report intermediate value for pruning
            if epoch % 5 == 0:  # Report every 5 epochs
                trial.report(val_accuracy, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Early stopping
            if no_improve_count >= patience:
                break
        
        training_time = time.time() - start_time
        return best_val_loss, best_val_acc, training_time
    
    def optimize(self, n_trials: Optional[int] = None, **kwargs):
        """
        Run device-optimized hyperparameter optimization.
        
        Args:
            n_trials: Number of trials to run (overrides config)
            **kwargs: Additional arguments for backward compatibility
        """
        n_trials = n_trials or self.opt_config['n_trials']
        
        print(f"\n🎯 Starting {self.optimization_strategy} optimization with {n_trials} trials...")
        print(f"📱 Device: {self.device}")
        
        if self.optimization_strategy == 'multiprocessing':
            self._cpu_multiprocess_optimize(n_trials)
        elif self.optimization_strategy == 'vectorized':
            self._gpu_vectorized_optimize(n_trials)
        else:
            # Fallback to sequential
            print("🔄 Running sequential optimization...")
            self.study.optimize(
                self.objective,
                n_trials=n_trials,
                n_jobs=1,
                callbacks=[display_progress_callback]
            )
        
        # Print results
        print("\n" + "="*60)
        print("🎉 OPTIMIZATION COMPLETED!")
        print("="*60)
        print(f"Strategy used: {self.optimization_strategy}")
        print(f"Device: {self.device}")
        print(f"Total trials: {len(self.study.trials)}")
        print(f"Best trial: #{self.study.best_trial.number}")
        print(f"Best validation accuracy: {self.study.best_value:.4f}")
        print("\n🏆 Best hyperparameters:")
        for key, value in self.study.best_params.items():
            if isinstance(value, float):
                print(f"   • {key}: {value:.6f}")
            else:
                print(f"   • {key}: {value}")
        print("="*60)
    
    def _cpu_multiprocess_optimize(self, n_trials: int):
        """CPU-based multiprocessing optimization (existing implementation)."""
        n_jobs = self.opt_config.get('cpu', {}).get('n_jobs', 4)
        print(f"🚀 Launching {n_jobs} parallel worker processes...")
        self._parallel_optimize(n_trials, n_jobs)
    
    def _gpu_vectorized_optimize(self, n_trials: int):
        """GPU-based vectorized optimization (new implementation)."""
        if not VMAP_AVAILABLE:
            print("⚠️  vmap not available. Falling back to sequential optimization on GPU.")
            self.study.optimize(
                self.objective,
                n_trials=n_trials,
                n_jobs=1,
                callbacks=[display_progress_callback]
            )
            return
        
        print("🔥 Starting GPU vectorized optimization...")
        
        # Initialize vectorized optimizer
        vectorized_optimizer = VectorizedOptimizer(
            self.base_config, 
            self._cached_data, 
            self.device, 
            self.opt_config
        )
        
        batch_size = self.opt_config.get('gpu', {}).get('vectorized_batch_size', 8)
        completed_trials = 0
        
        print(f"🎯 Processing {n_trials} trials in batches of {batch_size}")
        
        while completed_trials < n_trials:
            # Calculate remaining trials for this batch
            remaining_trials = n_trials - completed_trials
            current_batch_size = min(batch_size, remaining_trials)
            
            print(f"\n📦 Processing batch {completed_trials // batch_size + 1}: {current_batch_size} networks")
            
            # Sample hyperparameters for the current batch
            param_sets = []
            trials = []
            
            for _ in range(current_batch_size):
                # Create a new trial to sample parameters
                trial = self.study.ask()
                trials.append(trial)
                
                # Sample hyperparameters using the trial
                params = self._sample_hyperparameters_for_trial(trial)
                param_sets.append(params)
            
            # Train batch of networks
            try:
                # Determine number of epochs based on sampled parameters
                n_epochs = max([params.get('epochs', 50) for params in param_sets])
                
                print(f"   🏋️ Training {current_batch_size} networks for {n_epochs} epochs...")
                val_accuracies = vectorized_optimizer.train_batch(param_sets, n_epochs)
                
                # Report results back to Optuna
                for i, (trial, val_acc) in enumerate(zip(trials, val_accuracies)):
                    try:
                        # Tell Optuna the result for this trial
                        self.study.tell(trial, val_acc)
                        print(f"   ✅ Trial {trial.number}: {val_acc:.4f}")
                    except Exception as e:
                        print(f"   ❌ Failed to report trial {trial.number}: {e}")
                
                completed_trials += current_batch_size
                
                # Show progress
                progress_pct = (completed_trials / n_trials) * 100
                bar_length = 20
                filled_length = int(bar_length * completed_trials // n_trials)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"   📊 Progress: {bar} {progress_pct:.1f}% ({completed_trials}/{n_trials})")
                
                # Show current best
                if len(self.study.trials) > 0:
                    current_best = max([t.value for t in self.study.trials if t.value is not None])
                    print(f"   🏆 Best so far: {current_best:.4f}")
                
            except Exception as e:
                print(f"❌ Batch training failed: {e}")
                # Mark failed trials
                for trial in trials:
                    try:
                        self.study.tell(trial, 0.0)  # Report failure
                    except:
                        pass
                completed_trials += current_batch_size
    
    def _sample_hyperparameters_for_trial(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters for a single trial (used in vectorized mode)."""
        params = {}
        
        # Learning rate
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        
        # Batch size
        params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
        
        # Dropout rate
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.0, 0.5)
        
        # Regularization
        params['l1_lambda'] = trial.suggest_float('l1_lambda', 1e-6, 1e-1, log=True)
        params['l2_lambda'] = trial.suggest_float('l2_lambda', 1e-6, 1e-1, log=True)
        
        # Optimizer
        params['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        
        # Weight decay
        params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        # SGD specific parameters
        if params['optimizer'] == 'sgd':
            params['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
        
        # Gradient clipping
        use_grad_clip = trial.suggest_categorical('use_gradient_clipping', [True, False])
        if use_grad_clip:
            params['grad_clip_max_norm'] = trial.suggest_float('grad_clip_max_norm', 0.1, 5.0)
        
        # Number of epochs
        params['epochs'] = trial.suggest_int('epochs', 30, 100)
        
        return params
    
    def _parallel_optimize(self, n_trials: int, n_jobs: int):
        """
        Run optimization using true multiprocessing.
        """
        # Prepare arguments for workers
        worker_args = (
            self.opt_config.get('storage'),
            self.opt_config['study_name'],
            self.base_config,
            self._cached_data
        )
        
        completed_trials = 0
        
        try:
            # Use ProcessPoolExecutor for true parallelism
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn')) as executor:
                print(f"✅ Process pool started with {n_jobs} workers")
                
                # Submit initial batch of jobs
                futures = []
                for _ in range(min(n_jobs, n_trials)):
                    future = executor.submit(parallel_trial_worker, worker_args)
                    futures.append(future)
                
                # Process completed trials and submit new ones
                while completed_trials < n_trials and futures:
                    # Wait for at least one trial to complete
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=300)  # 5 minute timeout per trial
                            if result:
                                trial_num, trial_val, trial_params, trial_state = result
                                completed_trials += 1
                                
                                # Display progress
                                print(f"\n✅ Trial {trial_num} completed (Process {completed_trials}/{n_trials})")
                                if trial_val is not None:
                                    print(f"   • Validation Accuracy: {trial_val:.4f}")
                                    if completed_trials > 0:
                                        current_best = max([t.value for t in self.study.trials if t.value is not None])
                                        print(f"   • Best so far: {current_best:.4f}")
                                else:
                                    print(f"   • Trial was pruned or failed")
                                
                                # Show progress bar
                                progress_pct = (completed_trials / n_trials) * 100
                                bar_length = 20
                                filled_length = int(bar_length * completed_trials // n_trials)
                                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                                print(f"   • Progress: {bar} {progress_pct:.1f}%")
                            
                        except Exception as e:
                            print(f"❌ Trial failed: {e}")
                        
                        # Remove completed future
                        futures.remove(future)
                        
                        # Submit new trial if we haven't reached the limit
                        if completed_trials < n_trials:
                            new_future = executor.submit(parallel_trial_worker, worker_args)
                            futures.append(new_future)
                        
                        break  # Process one completion at a time
                        
        except KeyboardInterrupt:
            print("\n⚠️  Optimization interrupted by user")
        except Exception as e:
            print(f"\n❌ Parallel optimization failed: {e}")
            print("🔄 Falling back to sequential optimization...")
            remaining_trials = n_trials - completed_trials
            if remaining_trials > 0:
                self.study.optimize(
                    self.objective,
                    n_trials=remaining_trials,
                    n_jobs=1,
                    callbacks=[display_progress_callback]
                )
    
    def save_best_config(self, output_path: str = "configs/optimized_parameters.yaml"):
        """
        Save the best hyperparameters to a YAML file.
        
        Args:
            output_path: Path to save the optimized configuration
        """
        # Create optimized config based on best trial
        optimized_config = deepcopy(self.base_config)
        best_params = self.study.best_params
        
        # Update configuration with best parameters
        optimized_config['training']['learning_rate'] = best_params['learning_rate']
        optimized_config['training']['batch_size'] = best_params['batch_size']
        optimized_config['model']['dropout_rate'] = best_params['dropout_rate']
        
        optimized_config['training']['regularization'] = {
            'l1_lambda': best_params['l1_lambda'],
            'l2_lambda': best_params['l2_lambda']
        }
        
        optimized_config['training']['optimizer']['name'] = best_params['optimizer']
        optimized_config['training']['optimizer']['weight_decay'] = best_params['weight_decay']
        
        if best_params['optimizer'] == 'sgd' and 'momentum' in best_params:
            optimized_config['training']['optimizer']['momentum'] = best_params['momentum']
        
        if best_params.get('use_gradient_clipping', False):
            optimized_config['training']['gradient_clipping'] = {
                'enabled': True,
                'max_norm': best_params.get('grad_clip_max_norm', 1.0)
            }
        
        optimized_config['training']['epochs'] = best_params['epochs']
        
        # Add optimization metadata
        optimized_config['optimization_metadata'] = {
            'best_validation_accuracy': float(self.study.best_value),
            'best_validation_loss': float(self.study.best_trial.user_attrs.get('validation_loss', 0)),
            'best_trial_number': self.study.best_trial.number,
            'total_trials': len(self.study.trials),
            'optimization_date': datetime.now().isoformat(),
            'training_time': self.study.best_trial.user_attrs.get('training_time', 0)
        }
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\nOptimized configuration saved to: {output_path}")
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot the optimization history.
        
        Args:
            save_path: Path to save the plot (if None, just displays)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get trial data
            trials = self.study.trials
            trial_numbers = [t.number for t in trials if t.value is not None]
            trial_values = [t.value for t in trials if t.value is not None]
            
            # Plot optimization history
            plt.figure(figsize=(10, 6))
            plt.plot(trial_numbers, trial_values, 'bo-', alpha=0.6, label='Trial accuracy')
            
            # Add best value line
            best_value = max(trial_values) if trial_values else 0
            plt.axhline(y=best_value, color='r', linestyle='--', label=f'Best: {best_value:.4f}')
            
            plt.xlabel('Trial Number')
            plt.ylabel('Validation Accuracy')
            plt.title('Optimization History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Optimization history plot saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
    
    def plot_parameter_importance(self, save_path: Optional[str] = None):
        """
        Plot parameter importance.
        
        Args:
            save_path: Path to save the plot (if None, just displays)
        """
        try:
            import matplotlib.pyplot as plt
            from optuna.importance import get_param_importances
            
            # Calculate parameter importance
            importance = get_param_importances(self.study)
            
            # Create bar plot
            plt.figure(figsize=(10, 6))
            params = list(importance.keys())
            values = list(importance.values())
            
            plt.barh(params, values)
            plt.xlabel('Importance')
            plt.title('Hyperparameter Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Parameter importance plot saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
        except Exception as e:
            print(f"Could not generate parameter importance plot: {e}")


def print_banner():
    """Print a nice banner for the optimization tool."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║           Bayesian Hyperparameter Optimization                ║
║                    for torch_mlp                              ║
║                                                               ║
║  🖥️  CPU: Multiprocessing optimization                        ║
║  🚀 GPU: Vectorized training (CUDA/MPS)                       ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def get_user_choice(prompt: str, options: list, default: int = 0) -> int:
    """Get user choice from a list of options."""
    print(f"\n{prompt}")
    for i, option in enumerate(options):
        marker = " [default]" if i == default else ""
        print(f"  {i+1}. {option}{marker}")
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(options)}) or press Enter for default: ").strip()
            if not choice:
                return default
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                return choice_idx
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def get_numeric_input(prompt: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """Get numeric input from user with validation."""
    range_str = ""
    if min_val is not None and max_val is not None:
        range_str = f" (range: {min_val}-{max_val})"
    
    while True:
        try:
            value = input(f"\n{prompt}{range_str} [default: {default}]: ").strip()
            if not value:
                return default
            value = float(value)
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")


def interactive_setup():
    """Interactive setup for optimization parameters."""
    print_banner()
    
    print("\n🚀 Welcome to the Hyperparameter Optimization Tool!\n")
    print("This tool will help you find the best hyperparameters for your torch_mlp model.")
    print("⚡ Two optimization strategies:")
    print("   • CPU: Multiprocessing with parallel workers")
    print("   • GPU: Vectorized training of multiple networks simultaneously")
    print("Let's configure the optimization settings...\n")
    
    # Quick start or custom setup
    setup_options = [
        "Quick Start (recommended defaults)",
        "Custom Setup (configure all options)",
        "Load from optimization_config.yaml"
    ]
    setup_choice = get_user_choice("How would you like to proceed?", setup_options, default=0)
    
    if setup_choice == 0:  # Quick Start
        config = {
            'base_config': 'configs/training_config.yaml',
            'n_trials': 25,  # Reduced for faster testing
            'device': 'auto',  # Auto-detect best device
            'study_name': 'torch_mlp_optimization',
            'output_config': 'configs/optimized_parameters.yaml',
            'plot_history': True,
            'plot_importance': True
        }
        print("\n✅ Using recommended settings:")
        print(f"   • Trials: {config['n_trials']} (fast optimization)")
        print(f"   • Device: {config['device']} (automatic detection)")
        print(f"   • Output: {config['output_config']}")
        print("   • Strategy: CPU multiprocessing or GPU vectorized (auto-selected)")
        
    elif setup_choice == 1:  # Custom Setup
        config = {}
        
        # Base configuration
        config['base_config'] = input("\nBase training config path [configs/training_config.yaml]: ").strip()
        if not config['base_config']:
            config['base_config'] = 'configs/training_config.yaml'
        
        # Number of trials
        config['n_trials'] = int(get_numeric_input(
            "Number of optimization trials", 
            default=100, 
            min_val=10, 
            max_val=1000
        ))
        
        # Number of parallel jobs
        import multiprocessing
        max_cores = multiprocessing.cpu_count()
        config['n_jobs'] = int(get_numeric_input(
            f"Number of parallel jobs (you have {max_cores} CPU cores)", 
            default=min(4, max_cores), 
            min_val=1, 
            max_val=max_cores
        ))
        
        # Study name
        study_name = input("\nStudy name [torch_mlp_optimization]: ").strip()
        config['study_name'] = study_name if study_name else 'torch_mlp_optimization'
        
        # Output path
        output_path = input("\nOutput config path [configs/optimized_parameters.yaml]: ").strip()
        config['output_config'] = output_path if output_path else 'configs/optimized_parameters.yaml'
        
        # Plots
        plot_options = ["Yes", "No"]
        config['plot_history'] = get_user_choice("Generate optimization history plot?", plot_options, default=0) == 0
        config['plot_importance'] = get_user_choice("Generate parameter importance plot?", plot_options, default=0) == 0
        
    else:  # Load from config
        with open('configs/optimization_config.yaml', 'r') as f:
            opt_config = yaml.safe_load(f)['optimization']
        config = {
            'base_config': 'configs/training_config.yaml',
            'n_trials': opt_config['n_trials'],
            'device': opt_config.get('device', 'auto'),
            'n_jobs': opt_config.get('cpu', {}).get('n_jobs', 4),  # For backward compatibility
            'study_name': opt_config['study_name'],
            'output_config': 'configs/optimized_parameters.yaml',
            'plot_history': True,
            'plot_importance': True
        }
        print("\n✅ Loaded settings from optimization_config.yaml")
        print(f"   • Device: {config['device']}")
    
    # Confirmation
    print("\n" + "="*60)
    print("📋 OPTIMIZATION CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Base config:      {config['base_config']}")
    print(f"Number of trials: {config['n_trials']}")
    print(f"Device:           {config.get('device', 'auto')}")
    if config.get('device') == 'cpu' or config.get('device') == 'auto':
        print(f"CPU workers:      {config.get('n_jobs', 4)}")
    print(f"Study name:       {config['study_name']}")
    print(f"Output config:    {config['output_config']}")
    print(f"Generate plots:   History={config['plot_history']}, Importance={config['plot_importance']}")
    print("="*60)
    
    proceed_options = ["Start Optimization", "Cancel"]
    if get_user_choice("\nReady to start?", proceed_options, default=0) == 1:
        print("\n❌ Optimization cancelled.")
        return None
    
    return config


def display_progress_callback(study, trial):
    """Callback to display progress during optimization."""
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    total_trials = len(study.trials)
    
    if trial.state == optuna.trial.TrialState.COMPLETE:
        # Progress bar
        progress_pct = (n_complete / max(25, n_complete)) * 100  # Assume at least 25 trials
        bar_length = 20
        filled_length = int(bar_length * n_complete // max(25, n_complete))
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\n✅ Trial {trial.number} completed | {bar} {progress_pct:.1f}%")
        print(f"   • Validation Accuracy: {trial.value:.4f}")
        print(f"   • Best so far: {study.best_value:.4f}")
        print(f"   • Progress: {n_complete} completed, {n_pruned} pruned, {total_trials} total")
        
        # Show training time if available
        if 'training_time' in trial.user_attrs:
            print(f"   • Training time: {trial.user_attrs['training_time']:.1f}s")
        
        # Show current best parameters every 5 trials
        if n_complete % 5 == 0 and n_complete > 0:
            print("\n🏆 Current best parameters:")
            for key, value in study.best_params.items():
                if isinstance(value, float):
                    print(f"   • {key}: {value:.6f}")
                else:
                    print(f"   • {key}: {value}")
            print()
    
    elif trial.state == optuna.trial.TrialState.PRUNED:
        print(f"⚡ Trial {trial.number} pruned (early stopped)")
    elif trial.state == optuna.trial.TrialState.FAIL:
        print(f"❌ Trial {trial.number} failed")


def main():
    """
    Main function with interactive interface.
    """
    # Interactive setup
    config = interactive_setup()
    if config is None:
        return
    
    print("\n🔄 Starting optimization...\n")
    
    # Create optimization configuration
    # Load GPU config from the original config file if available
    try:
        with open('configs/optimization_config.yaml', 'r') as f:
            file_config = yaml.safe_load(f)['optimization']
            gpu_config = file_config.get('gpu', {})
    except:
        gpu_config = {}
    
    opt_config = {
        'n_trials': config['n_trials'],
        'device': config.get('device', 'auto'),
        'cpu': {'n_jobs': config.get('n_jobs', 4)},  # For backward compatibility
        'gpu': {
            'vectorized_batch_size': gpu_config.get('vectorized_batch_size', 8),
            'use_mixed_precision': gpu_config.get('use_mixed_precision', True),
            'max_memory_fraction': gpu_config.get('max_memory_fraction', 0.8)
        },
        'study_name': config['study_name'],
        'storage': f"sqlite:///{config['study_name']}.db"
    }
    
    print(f"🔧 GPU Config: Batch size = {opt_config['gpu']['vectorized_batch_size']}")
    
    # Check if study already exists
    try:
        existing_study = optuna.load_study(
            study_name=config['study_name'],
            storage=opt_config['storage']
        )
        n_existing = len(existing_study.trials)
        if n_existing > 0:
            print(f"⚠️  Found existing study with {n_existing} trials.")
            resume_options = ["Resume from existing study", "Start fresh (delete existing)"]
            if get_user_choice("What would you like to do?", resume_options, default=0) == 1:
                optuna.delete_study(
                    study_name=config['study_name'],
                    storage=opt_config['storage']
                )
                print("🗑️  Existing study deleted.")
    except KeyError:
        pass  # Study doesn't exist yet
    
    # Initialize optimizer
    print(f"\n📊 Initializing optimizer with {config['n_jobs']} parallel workers...")
    optimizer = HyperparameterOptimizer(config['base_config'], opt_config)
    
    # Run optimization with proper parallel/sequential handling
    optimizer.optimize(
        n_trials=config['n_trials'],
        n_jobs=config['n_jobs']
    )
    
    # Print final results
    print("\n" + "="*60)
    print("🎉 OPTIMIZATION COMPLETED!")
    print("="*60)
    print(f"Total trials run: {len(optimizer.study.trials)}")
    print(f"Best trial: #{optimizer.study.best_trial.number}")
    print(f"Best validation accuracy: {optimizer.study.best_value:.4f}")
    print("\n🏆 Best hyperparameters:")
    for key, value in optimizer.study.best_params.items():
        print(f"   • {key}: {value:.6f if isinstance(value, float) else value}")
    print("="*60)
    
    # Save best configuration
    print(f"\n💾 Saving optimized configuration to: {config['output_config']}")
    optimizer.save_best_config(config['output_config'])
    
    # Generate plots
    if config['plot_history'] or config['plot_importance']:
        print("\n📈 Generating visualization plots...")
        Path('results/optimization').mkdir(parents=True, exist_ok=True)
        
        if config['plot_history']:
            optimizer.plot_optimization_history('results/optimization/history.png')
        
        if config['plot_importance']:
            optimizer.plot_parameter_importance('results/optimization/importance.png')
    
    print("\n✨ All done! You can now:")
    print(f"   1. Use the optimized config: python src/models/torch_mlp.py {config['output_config']}")
    print(f"   2. View interactive dashboard: optuna-dashboard {opt_config['storage']}")
    print(f"   3. Check plots in: results/optimization/")
    print("\n📊 Pro Tip: For real-time monitoring during future runs:")
    print(f"   • Run optimization in one terminal")
    print(f"   • In another terminal: optuna-dashboard {opt_config['storage']}")
    print(f"   • Open browser to: http://localhost:8080")
    print("\nHappy training! 🚀")


if __name__ == "__main__":
    main()