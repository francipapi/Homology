"""
Trainer with Network Homology Tracking

This module extends the existing training pipeline to include network homology
tracking during training. It provides a wrapper that can be used with any of
the existing model types (MLP, custom architectures, etc.).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from torch.utils.data import DataLoader, TensorDataset

# Import existing modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.torch_mlp import generate_torus_data, load_data_from_file
from src.models.torch_custom import CustomNet
from src.topology.network_homology_tracker import NetworkHomologyTracker


class TrainerWithHomology:
    """
    Extended trainer that includes network homology tracking.
    
    This class wraps existing model training with homology computation
    at specified intervals.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], 
                 homology_config: Optional[Dict[str, Any]] = None):
        """
        Initialize trainer with homology tracking.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            homology_config: Network homology configuration (optional)
        """
        self.model = model
        self.config = config
        self.training_config = config['training']
        self.data_config = config['data']
        
        # Load homology configuration
        if homology_config is None:
            # Try to load from file
            homology_config_path = Path(__file__).parent.parent.parent / "configs" / "network_homology_config.yaml"
            if homology_config_path.exists():
                with open(homology_config_path, 'r') as f:
                    homology_config = yaml.safe_load(f)
        
        self.homology_config = homology_config
        self.homology_enabled = homology_config.get('network_homology', {}).get('enabled', False)
        
        # Initialize homology tracker if enabled
        if self.homology_enabled:
            self.homology_tracker = NetworkHomologyTracker(homology_config)
            
            # Get alignment settings from simplified configuration
            alignment_config = homology_config.get('network_homology', {}).get('alignment', {})
            
            # Fallback to main config if not found in homology config
            if not alignment_config and config.get('alignment'):
                alignment_config = config.get('alignment', {})
            
            self.track_mode = str(alignment_config.get('mode', 'epoch'))  # 'epoch' or 'step'
            self.validation_interval = int(alignment_config.get('validation_interval', 1))  # Every N epochs/steps
            
            print(f"Network homology tracking enabled")
            print(f"  Mode: {self.track_mode} (track every {self.validation_interval} {self.track_mode}s)")
            print(f"  Perfect alignment: validation and homology measured together")
        else:
            self.homology_tracker = None
            print("Network homology tracking disabled")
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup loss function
        self.criterion = self._setup_criterion()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_validation_accuracy = 0.0
        
        # Perfect alignment tracking
        self.synchronized_measurements = []  # Store (step, epoch, homology_distance, validation_accuracy)
        self.current_test_loader = None  # Store test loader for step-based validation
        
    def _setup_device(self) -> torch.device:
        """Setup and return the appropriate device."""
        device_name = self.training_config.get('device', 'cpu')
        
        if device_name == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        elif device_name == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        elif device_name == 'mps' and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(device_name)
        
        print(f"Using device: {device}")
        return device
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup and return the optimizer."""
        lr = self.training_config['learning_rate']
        opt_config = self.training_config.get('optimizer', {'type': 'adam'})
        optimizer_type = opt_config.get('type', opt_config.get('name', 'adam')).lower()
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, 
                            weight_decay=opt_config.get('weight_decay', 0.0))
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, 
                             weight_decay=opt_config.get('weight_decay', 0.01))
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, 
                           weight_decay=opt_config.get('weight_decay', 0.0),
                           momentum=opt_config.get('momentum', 0.9))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _setup_criterion(self) -> nn.Module:
        """Setup and return the loss function."""
        loss_fn = self.training_config.get('loss_fn', 'bce')
        
        if loss_fn == 'bce':
            return nn.BCELoss()
        elif loss_fn == 'mse':
            return nn.MSELoss()
        elif loss_fn == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        # Load or generate data
        data_source = self.data_config.get('data_source')
        
        if data_source is not None:
            print(f"Loading data from: {data_source}")
            X, y = load_data_from_file(data_source)
        elif self.data_config['type'] == 'synthetic':
            num_samples = self.data_config.get('generation', {}).get('n', 1000)
            big_radius = self.data_config.get('generation', {}).get('big_radius', 3)
            small_radius = self.data_config.get('generation', {}).get('small_radius', 1)
            solid = self.data_config.get('generation', {}).get('solid', False)
            interior_noise = self.data_config.get('generation', {}).get('interior_noise', 0.1)
            X, y = generate_torus_data(num_samples, big_radius, small_radius, solid, interior_noise)
        else:
            raise ValueError("Either set data_source or use synthetic data")
        
        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Shuffle data
        perm = torch.randperm(len(X))
        X = X[perm]
        y = y[perm]
        
        # Split data
        split_ratio = self.data_config.get('split_ratio', 0.8)
        train_size = int(split_ratio * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        batch_size = self.training_config['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, test_loader: DataLoader = None) -> Dict[str, float]:
        """Train for one epoch and track homology if enabled."""
        # Store test loader for step-based validation access
        self.current_test_loader = test_loader
        
        self.model.train()
        train_loss_sum = 0
        correct_train = 0
        total_train = 0
        
        epoch_start_time = time.time()
        homology_times = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Regular training step
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Ensure output shape matches target shape
            if output.shape != target.shape:
                output = output.squeeze(-1)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            train_loss_sum += loss.item()
            predicted = (output > 0.5).float()
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # Perfect alignment tracking - only for step-based mode
            # Fix: Skip step 0 to avoid incorrect counting
            should_track_step = (self.homology_enabled and 
                               self.track_mode == 'step' and 
                               self.global_step > 0 and 
                               self.global_step % self.validation_interval == 0)
            
            if should_track_step:
                self._perform_synchronized_measurement(
                    epoch, batch_idx, loss.item(), train_loader, homology_times
                )
            
            self.global_step += 1
        
        # Calculate epoch metrics
        avg_train_loss = train_loss_sum / len(train_loader)
        train_accuracy = correct_train / total_train
        epoch_time = time.time() - epoch_start_time
        
        metrics = {
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'epoch_time': epoch_time
        }
        
        # Perfect alignment tracking - epoch-based mode
        should_track_epoch = (self.homology_enabled and 
                            self.track_mode == 'epoch' and 
                            (epoch + 1) % self.validation_interval == 0)
        
        if should_track_epoch:
            # Perform synchronized measurement at end of epoch
            validation_accuracy = self._compute_validation_accuracy(test_loader)
            self._perform_synchronized_measurement(
                epoch, len(train_loader), avg_train_loss, None, homology_times,
                validation_accuracy=validation_accuracy
            )
        
        if homology_times:
            metrics['avg_homology_time'] = np.mean(homology_times)
            metrics['total_homology_time'] = sum(homology_times)
        
        return metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        test_loss_sum = 0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                if output.shape != target.shape:
                    output = output.squeeze(-1)
                
                loss = self.criterion(output, target)
                test_loss_sum += loss.item()
                
                predicted = (output > 0.5).float()
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
        
        avg_test_loss = test_loss_sum / len(test_loader)
        test_accuracy = correct_test / total_test
        
        return {
            'test_loss': avg_test_loss,
            'test_accuracy': test_accuracy
        }
    
    def _quick_validation(self, dataset) -> float:
        """Quick validation accuracy computation for homology tracking."""
        self.model.eval()
        
        # Sample a subset of the validation data
        sample_size = min(256, len(dataset))
        indices = torch.randperm(len(dataset))[:sample_size]
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for idx in indices:
                data, target = dataset[idx]
                data = data.unsqueeze(0).to(self.device)
                target = target.unsqueeze(0).to(self.device)
                
                output = self.model(data)
                if output.shape != target.shape:
                    output = output.squeeze(-1)
                
                predicted = (output > 0.5).float()
                correct += (predicted == target).sum().item()
                total += 1
        
        self.model.train()
        return correct / total if total > 0 else 0.0
    
    def _compute_step_validation_accuracy(self) -> float:
        """
        Compute validation accuracy during step-based tracking.
        
        This implements proper validation on test set during training steps,
        following nn-evolution's approach but with better temporal alignment.
        
        Returns:
            Validation accuracy on test set
        """
        if self.current_test_loader is None:
            # Fallback to quick validation if no test loader available
            return 0.0
        
        self.model.eval()
        correct = 0
        total = 0
        
        # Sample a subset of test data for efficiency during step-based tracking
        max_samples = self.homology_config.get('alignment', {}).get('quick_validation_samples', 256)
        samples_processed = 0
        
        with torch.no_grad():
            for data, target in self.current_test_loader:
                # Stop if we've processed enough samples
                if samples_processed >= max_samples:
                    break
                
                output = self.model(data)
                if output.shape != target.shape:
                    output = output.squeeze(-1)
                
                predicted = (output > 0.5).float()
                correct += (predicted == target).sum().item()
                total += target.size(0)
                samples_processed += target.size(0)
        
        self.model.train()
        return correct / total if total > 0 else 0.0
    
    def _compute_validation_accuracy(self, test_loader: DataLoader = None) -> float:
        """
        Compute validation accuracy for synchronized measurements.
        
        Args:
            test_loader: Test data loader (if None, uses quick validation)
            
        Returns:
            Validation accuracy
        """
        if test_loader is None:
            # Fallback to quick validation on training data
            return self._quick_validation(None)
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                if output.shape != target.shape:
                    output = output.squeeze(-1)
                
                predicted = (output > 0.5).float()
                correct += (predicted == target).sum().item()
                total += target.size(0)
        
        self.model.train()
        return correct / total if total > 0 else 0.0
    
    def _perform_synchronized_measurement(self, epoch: int, batch_idx: int, 
                                        train_loss: float, train_loader: DataLoader = None,
                                        homology_times: list = None, 
                                        validation_accuracy: float = None) -> None:
        """
        Perform perfectly synchronized homology and validation measurements.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            train_loss: Current training loss
            train_loader: Training data loader (for step-based validation)
            homology_times: List to append computation times
            validation_accuracy: Pre-computed validation accuracy (for epoch-based)
        """
        homology_start = time.time()
        
        # Compute validation accuracy if not provided
        if validation_accuracy is None:
            if self.track_mode == 'step':
                # Step-based: use proper validation on test set
                validation_accuracy = self._compute_step_validation_accuracy()
            elif train_loader is not None:
                # Fallback: quick validation on training set (for epoch mode only)
                validation_accuracy = self._quick_validation(train_loader.dataset)
            else:
                # This shouldn't happen, but fallback
                validation_accuracy = 0.0
        
        # Track homology with synchronized validation measurement
        distance, snapshot = self.homology_tracker.track_training_step(
            model=self.model,
            step=self.global_step,
            epoch=epoch,
            batch_idx=batch_idx,
            validation_accuracy=validation_accuracy,
            train_loss=train_loss
        )
        
        homology_time = time.time() - homology_start
        if homology_times is not None:
            homology_times.append(homology_time)
        
        # Store synchronized measurement
        self.synchronized_measurements.append({
            'step': self.global_step,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'homology_distance': distance,
            'validation_accuracy': validation_accuracy,
            'train_loss': train_loss,
            'computation_time': homology_time
        })
        
        # Print progress with clearer formatting
        mode_str = f"Epoch {epoch+1}" if self.track_mode == 'epoch' else f"Step {self.global_step}"
        print(f"  📊 {mode_str}: Homology distance = {distance:.4f}, "
              f"Validation accuracy = {validation_accuracy:.4f} "
              f"(computed in {homology_time:.2f}s)")
    
    def get_synchronized_data(self) -> Dict[str, np.ndarray]:
        """
        Get perfectly aligned homology distances and validation accuracies.
        
        Returns:
            Dictionary with aligned arrays for correlation computation
        """
        if not self.synchronized_measurements:
            return {'distances': np.array([]), 'validations': np.array([])}
        
        distances = [m['homology_distance'] for m in self.synchronized_measurements]
        validations = [m['validation_accuracy'] for m in self.synchronized_measurements]
        
        return {
            'distances': np.array(distances),
            'validations': np.array(validations),
            'steps': np.array([m['step'] for m in self.synchronized_measurements]),
            'epochs': np.array([m['epoch'] for m in self.synchronized_measurements]),
            'train_losses': np.array([m['train_loss'] for m in self.synchronized_measurements])
        }
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Main training loop with homology tracking.
        
        Args:
            num_epochs: Number of epochs to train (if None, uses config)
            
        Returns:
            Dictionary with training results and statistics
        """
        if num_epochs is None:
            num_epochs = self.training_config['epochs']
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Model architecture:\n{self.model}")
        
        # Prepare data
        train_loader, test_loader = self.prepare_data()
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'epoch_times': []
        }
        
        # Main training loop
        total_start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch, test_loader)
            
            # Evaluate
            test_metrics = self.evaluate(test_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_accuracy'].append(train_metrics['train_accuracy'])
            history['test_loss'].append(test_metrics['test_loss'])
            history['test_accuracy'].append(test_metrics['test_accuracy'])
            history['epoch_times'].append(train_metrics['epoch_time'])
            
            # Track best model
            if test_metrics['test_accuracy'] > self.best_validation_accuracy:
                self.best_validation_accuracy = test_metrics['test_accuracy']
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_accuracy']:.4f} - "
                  f"Test Loss: {test_metrics['test_loss']:.4f}, "
                  f"Test Acc: {test_metrics['test_accuracy']:.4f}")
            
            if 'avg_homology_time' in train_metrics:
                print(f"  Homology computation: {train_metrics['avg_homology_time']:.2f}s avg, "
                      f"{train_metrics['total_homology_time']:.2f}s total")
        
        total_time = time.time() - total_start_time
        
        # Final results
        results = {
            'history': history,
            'best_validation_accuracy': self.best_validation_accuracy,
            'total_training_time': total_time,
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_test_accuracy': history['test_accuracy'][-1]
        }
        
        # Add homology results if enabled
        if self.homology_enabled and self.homology_tracker:
            homology_stats = self.homology_tracker.get_summary_statistics()
            results['homology_statistics'] = homology_stats
            
            # Get synchronized data for perfect alignment
            sync_data = self.get_synchronized_data()
            results['synchronized_data'] = sync_data
            
            # Compute correlation using perfectly aligned data
            if len(sync_data['distances']) > 1 and len(sync_data['validations']) > 1:
                # Use cumulative distances (nn-evolution style)
                cumulative_distances = np.cumsum(sync_data['distances'])
                
                # Compute correlation
                try:
                    from scipy.stats import pearsonr
                    correlation, p_value = pearsonr(cumulative_distances, sync_data['validations'])
                    results['synchronized_correlation'] = correlation
                    results['correlation_p_value'] = p_value
                except ImportError:
                    correlation = np.corrcoef(cumulative_distances, sync_data['validations'])[0, 1]
                    results['synchronized_correlation'] = correlation
                    p_value = None
                
                # Also compute with original method for comparison
                original_correlation = self.homology_tracker.compute_correlation_with_validation(
                    use_nn_evolution_style=False
                )
                results['original_correlation'] = original_correlation
                
                # Compute with nn-evolution's exact methodology
                nn_evolution_correlation = self.homology_tracker.compute_correlation_with_validation(
                    use_nn_evolution_style=True
                )
                results['nn_evolution_correlation'] = nn_evolution_correlation
            else:
                results['synchronized_correlation'] = 0.0
                correlation = 0.0
                p_value = None
                results['original_correlation'] = 0.0
            
            print(f"\n🔬 Homology tracking complete:")
            print(f"  📈 Total homology computations: {homology_stats['num_snapshots']}")
            print(f"  🎯 Synchronized measurements: {len(sync_data['distances'])}")
            print(f"  📊 Tracking mode: {self.track_mode} (every {self.validation_interval} {self.track_mode}s)")
            print(f"  🔗 Perfect alignment correlation: {results.get('synchronized_correlation', 0.0):.4f}")
            print(f"  🔬 nn-evolution style correlation: {results.get('nn_evolution_correlation', 0.0):.4f}")
            print(f"  📊 Original correlation: {results.get('original_correlation', 0.0):.4f}")
            if p_value is not None:
                print(f"  📊 P-value: {p_value:.6f}")
            print(f"  ⏱️  Average computation time: {homology_stats['average_computation_time']:.2f}s")
        
        print(f"\nTraining completed in {total_time:.2f} seconds")
        
        return results
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save all training and homology results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / "model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_validation_accuracy': self.best_validation_accuracy
        }, model_path)
        
        print(f"Model saved to {model_path}")
        
        # Save homology results if enabled
        if self.homology_enabled and self.homology_tracker:
            homology_dir = output_dir / "network_homology"
            self.homology_tracker.save_results(homology_dir)
            print(f"Homology results saved to {homology_dir}")
            
            # Create network visualizations based on configuration
            viz_config = self.homology_config.get('network_homology', {}).get('visualization', {})
            viz_enabled = bool(viz_config.get('enabled', True))
            
            if viz_enabled:
                from src.visualization.network_graph_viz import NetworkGraphVisualizer
                visualizer = NetworkGraphVisualizer()
                
                # Static visualization
                create_static = bool(viz_config.get('create_static_graph', True))
                if create_static:
                    static_format = str(viz_config.get('static_format', 'png'))
                    static_dpi = int(viz_config.get('static_dpi', 300))
                    static_path = homology_dir / f"network_graph.{static_format}"
                    
                    print(f"Creating static graph visualization: {static_path}")
                    # Set matplotlib DPI for static visualization
                    import matplotlib.pyplot as plt
                    original_dpi = plt.rcParams.get('figure.dpi', 100)
                    plt.rcParams['figure.dpi'] = static_dpi
                    
                    try:
                        visualizer.visualize_network(
                            self.model, 
                            method='static',
                            save_path=str(static_path)
                        )
                    finally:
                        # Restore original DPI
                        plt.rcParams['figure.dpi'] = original_dpi
                
                # Interactive visualization
                create_interactive = bool(viz_config.get('create_interactive_graph', True))
                if create_interactive:
                    interactive_format = str(viz_config.get('interactive_format', 'html'))
                    interactive_path = homology_dir / f"network_graph.{interactive_format}"
                    
                    print(f"Creating interactive graph visualization: {interactive_path}")
                    visualizer.visualize_network(
                        self.model,
                        method='interactive',
                        save_path=str(interactive_path)
                    )
            else:
                print("Network visualization disabled in configuration")


def train_with_homology(training_config_path: str, 
                       homology_config_path: Optional[str] = None,
                       model_type: str = "mlp") -> Dict[str, Any]:
    """
    Convenience function to train a model with homology tracking.
    
    Args:
        training_config_path: Path to training configuration
        homology_config_path: Path to homology configuration (optional)
        model_type: Type of model to train ("mlp" or "custom")
        
    Returns:
        Training results dictionary
    """
    # Load configurations
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    homology_config = None
    if homology_config_path:
        with open(homology_config_path, 'r') as f:
            homology_config = yaml.safe_load(f)
    
    # Create model
    if model_type == "custom":
        if not training_config.get('custom_architecture', {}).get('enabled', False):
            raise ValueError("Custom architecture not enabled in config")
        model = CustomNet(training_config['custom_architecture'])
    else:
        # Default MLP
        from src.models.torch_mlp import MLP
        model_config = training_config['model']
        model = MLP(
            input_dim=model_config['input_dim'],
            num_hidden_layers=model_config['num_hidden_layers'],
            hidden_dim=model_config['hidden_dim'],
            output_dim=model_config['output_dim'],
            activation_fn_name=model_config['activation_fn_name'],
            dropout_rate=model_config.get('dropout_rate', 0.0),
            use_batch_norm=model_config.get('use_batch_norm', False)
        )
        # Add input_shape attribute for graph builder
        model.input_shape = (model_config['input_dim'],)
    
    # Create trainer
    trainer = TrainerWithHomology(model, training_config, homology_config)
    
    # Train
    results = trainer.train()
    
    # Save results
    output_dir = Path("results") / "homology_training" / f"{model_type}_{int(time.time())}"
    trainer.save_results(output_dir)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model with network homology tracking")
    parser.add_argument("--training-config", type=str, default="configs/training_config.yaml",
                       help="Path to training configuration")
    parser.add_argument("--homology-config", type=str, default="configs/network_homology_config.yaml",
                       help="Path to homology configuration")
    parser.add_argument("--model-type", type=str, default="custom",
                       choices=["mlp", "custom"], help="Type of model to train")
    
    args = parser.parse_args()
    
    results = train_with_homology(
        args.training_config,
        args.homology_config,
        args.model_type
    )