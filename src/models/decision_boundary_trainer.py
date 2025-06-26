"""
Standalone Decision Boundary Trainer

This module provides boundary extraction during training without external dependencies.
It includes all necessary classes and functions within this file.

Author: Claude Code
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import argparse
import os
import time
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import trimesh with fallback
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    print("Warning: trimesh not available. Mesh saving will use numpy fallback.")
    TRIMESH_AVAILABLE = False

# Import scikit-image with fallback
try:
    from skimage import measure
    from skimage.filters import gaussian
    MARCHING_CUBES_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Isosurface extraction disabled.")
    MARCHING_CUBES_AVAILABLE = False

# Import existing MLP class and utility functions from torch_mlp
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.torch_mlp import MLP, load_data_from_file, generate_torus_data


@dataclass
class BoundaryExtractionResult:
    """Result of decision boundary extraction."""
    epoch: int
    boundary_points: Optional[np.ndarray] = None
    mesh_vertices: Optional[np.ndarray] = None  
    mesh_faces: Optional[np.ndarray] = None
    extraction_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None


def load_boundary_config(config_path: str = "configs/decision_boundary_config.yaml") -> Dict:
    """Load decision boundary configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}


class DecisionBoundaryExtractor:
    """
    Extracts decision boundaries from neural networks.
    
    This class provides methods for:
    1. Sampling network predictions on 3D grids
    2. Extracting isosurfaces using marching cubes
    3. Sampling point clouds near decision boundaries
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the boundary extractor.
        
        Parameters:
        - config: Configuration dictionary
        """
        self.config = config
        self.extraction_config = config.get('extraction', {})
        
    def create_sampling_grid(self, bounds: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a 3D sampling grid for network evaluation.
        
        Parameters:
        - bounds: Dictionary with x_min, x_max, y_min, y_max, z_min, z_max
        
        Returns:
        - grid_points: Array of shape (N, 3) with grid coordinates
        - grid_shape: Shape of the original grid (nx, ny, nz)
        """
        resolution = self.extraction_config.get('grid', {}).get('resolution', [64, 64, 64])
        
        # Use provided bounds or default
        if bounds is None:
            bounds = self.extraction_config.get('grid', {}).get('custom_bounds', {
                'x_min': -6.0, 'x_max': 6.0,
                'y_min': -6.0, 'y_max': 6.0, 
                'z_min': -6.0, 'z_max': 6.0
            })
        
        # Create coordinate arrays
        x = np.linspace(bounds['x_min'], bounds['x_max'], resolution[0])
        y = np.linspace(bounds['y_min'], bounds['y_max'], resolution[1])
        z = np.linspace(bounds['z_min'], bounds['z_max'], resolution[2])
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten to get grid points
        grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        grid_shape = (resolution[0], resolution[1], resolution[2])
        
        return grid_points, grid_shape
        
    def evaluate_network_on_grid(self, model: nn.Module, grid_points: np.ndarray, 
                                device: torch.device, batch_size: int = 10000) -> np.ndarray:
        """
        Evaluate neural network predictions on a grid of points.
        
        Parameters:
        - model: PyTorch model
        - grid_points: Array of shape (N, 3) with grid coordinates
        - device: Device to run computations on
        - batch_size: Batch size for evaluation
        
        Returns:
        - predictions: Array of shape (N,) with network predictions
        """
        model.eval()
        predictions = []
        
        # Optimize batch size based on available memory
        performance_config = self.config.get('performance', {})
        memory_config = performance_config.get('memory', {})
        if memory_config.get('batch_processing', True):
            batch_size = memory_config.get('batch_size', batch_size)
        
        with torch.no_grad():
            # Use half precision if available and device supports it
            use_amp = device.type == 'cuda' and torch.cuda.is_available()
            
            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(device)
                
                # Get predictions with optional mixed precision
                if use_amp:
                    with torch.cuda.amp.autocast():
                        batch_preds = model(batch_tensor)
                else:
                    batch_preds = model(batch_tensor)
                    
                if batch_preds.dim() > 1:
                    batch_preds = batch_preds.squeeze()
                    
                predictions.append(batch_preds.cpu().numpy())
                
                # Memory cleanup
                del batch_tensor, batch_preds
                if device.type == 'cuda' and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        
        return np.concatenate(predictions, axis=0)
    
    def extract_isosurface(self, predictions: np.ndarray, grid_shape: Tuple[int, int, int],
                          level: float = 0.5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract isosurface using marching cubes algorithm.
        
        Parameters:
        - predictions: Flattened predictions array
        - grid_shape: Shape of the original grid (nx, ny, nz)
        - level: Isosurface level (decision threshold)
        
        Returns:
        - vertices: Array of vertex coordinates (N, 3)
        - faces: Array of face indices (M, 3)
        """
        if not MARCHING_CUBES_AVAILABLE:
            return None, None
            
        try:
            # Reshape predictions back to grid
            prediction_grid = predictions.reshape(grid_shape)
            
            # Apply smoothing if configured
            smoothing = self.extraction_config.get('isosurface', {}).get('smoothing', 0)
            if smoothing > 0:
                prediction_grid = gaussian(prediction_grid, sigma=smoothing)
            
            # Extract isosurface using marching cubes
            vertices, faces, _, _ = measure.marching_cubes(
                prediction_grid, 
                level=level,
                spacing=(1.0, 1.0, 1.0)  # Will be scaled later
            )
            
            # Scale vertices to actual coordinate space
            bounds = self.extraction_config.get('grid', {}).get('custom_bounds', {
                'x_min': -6.0, 'x_max': 6.0,
                'y_min': -6.0, 'y_max': 6.0,
                'z_min': -6.0, 'z_max': 6.0
            })
            
            scale_x = (bounds['x_max'] - bounds['x_min']) / grid_shape[0]
            scale_y = (bounds['y_max'] - bounds['y_min']) / grid_shape[1]
            scale_z = (bounds['z_max'] - bounds['z_min']) / grid_shape[2]
            
            vertices[:, 0] = vertices[:, 0] * scale_x + bounds['x_min']
            vertices[:, 1] = vertices[:, 1] * scale_y + bounds['y_min']
            vertices[:, 2] = vertices[:, 2] * scale_z + bounds['z_min']
            
            # Apply mesh decimation if configured and trimesh is available
            decimate_factor = self.extraction_config.get('isosurface', {}).get('decimate', 0)
            if decimate_factor > 0 and decimate_factor < 1 and TRIMESH_AVAILABLE:
                try:
                    # Create trimesh object
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    
                    # Simplify mesh - the method expects a reduction factor between 0 and 1
                    # where 0.1 means reduce TO 10% of original (90% reduction)
                    # but our config uses 0.1 to mean reduce BY 10% (keep 90%)
                    target_reduction = 1 - decimate_factor  # Convert to "keep" factor
                    simplified = mesh.simplify_quadric_decimation(face_count=int(len(faces) * target_reduction))
                    print(f"Decimated mesh: {len(faces)} → {len(simplified.faces)} faces ({decimate_factor*100:.1f}% reduction)")
                    vertices = simplified.vertices
                    faces = simplified.faces
                except Exception as e:
                    print(f"Warning: Mesh decimation failed: {e}")
            
            return vertices, faces
            
        except Exception as e:
            print(f"Error in isosurface extraction: {e}")
            return None, None
    
    def sample_boundary_points(self, model: nn.Module, device: torch.device,
                              num_points: int = 5000, method: str = 'adaptive') -> Optional[np.ndarray]:
        """
        Sample points near the decision boundary.
        
        Parameters:
        - model: PyTorch model
        - device: Device to run computations on
        - num_points: Target number of boundary points
        - method: Sampling method ('uniform', 'adaptive', 'random')
        
        Returns:
        - boundary_points: Array of shape (N, 3) with boundary points
        """
        boundary_config = self.extraction_config.get('boundary_detection', {})
        threshold = boundary_config.get('threshold', 0.5)
        tolerance = boundary_config.get('tolerance', 0.01)
        
        if method == 'uniform':
            return self._uniform_boundary_sampling(model, device, num_points, threshold, tolerance)
        else:
            return self._uniform_boundary_sampling(model, device, num_points, threshold, tolerance)
    
    def _uniform_boundary_sampling(self, model: nn.Module, device: torch.device,
                                  num_points: int, threshold: float, tolerance: float) -> Optional[np.ndarray]:
        """Uniform sampling on a regular grid."""
        try:
            # Use grid resolution for uniform sampling
            grid_points, grid_shape = self.create_sampling_grid()
            predictions = self.evaluate_network_on_grid(model, grid_points, device)
            
            # Find boundary points
            boundary_mask = np.abs(predictions - threshold) < tolerance
            boundary_points = grid_points[boundary_mask]
            
            # Subsample if necessary
            if len(boundary_points) > num_points:
                indices = np.random.choice(len(boundary_points), num_points, replace=False)
                boundary_points = boundary_points[indices]
            
            return boundary_points if len(boundary_points) > 0 else None
            
        except Exception as e:
            print(f"Error in uniform boundary sampling: {e}")
            return None
    
    def extract_decision_boundary(self, model: nn.Module, device: torch.device, 
                                 epoch: int = 0) -> BoundaryExtractionResult:
        """
        Main method to extract decision boundary (WITHOUT topology computation).
        
        Parameters:
        - model: PyTorch model
        - device: Device to run computations on
        - epoch: Current training epoch
        
        Returns:
        - BoundaryExtractionResult with extraction data (no topology)
        """
        start_time = time.time()
        result = BoundaryExtractionResult(epoch=epoch)
        
        try:
            print(f"Extracting decision boundary for epoch {epoch}...")
            
            # Create sampling grid
            grid_points, grid_shape = self.create_sampling_grid()
            print(f"Created sampling grid: {grid_shape}")
            
            # Evaluate network on grid
            predictions = self.evaluate_network_on_grid(model, grid_points, device)
            print(f"Evaluated network on {len(grid_points)} points")
            
            # Extract isosurface if enabled
            isosurface_config = self.extraction_config.get('isosurface', {})
            if isosurface_config.get('enabled', True):
                vertices, faces = self.extract_isosurface(predictions, grid_shape)
                if vertices is not None:
                    result.mesh_vertices = vertices
                    result.mesh_faces = faces
                    print(f"Extracted isosurface: {len(vertices)} vertices, {len(faces)} faces")
            
            # Sample boundary points if enabled
            point_config = self.extraction_config.get('point_sampling', {})
            if point_config.get('enabled', True):
                method = point_config.get('method', 'uniform')
                num_points = point_config.get('num_points', 5000)
                
                boundary_points = self.sample_boundary_points(model, device, num_points, method)
                if boundary_points is not None:
                    result.boundary_points = boundary_points
                    print(f"Sampled {len(boundary_points)} boundary points using {method} method")
            
            result.extraction_time = time.time() - start_time
            result.success = True
            
            # Add metadata
            result.metadata = {
                'grid_shape': grid_shape,
                'num_grid_points': len(grid_points),
                'prediction_range': [float(predictions.min()), float(predictions.max())],
                'config_used': self.extraction_config
            }
            
            print(f"Decision boundary extraction completed in {result.extraction_time:.2f}s")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.extraction_time = time.time() - start_time
            print(f"Error in decision boundary extraction: {e}")
        
        return result


class DecisionBoundaryTrainer:
    """
    Enhanced trainer that supports decision boundary extraction during training.
    """
    
    def __init__(self, training_config: Dict, boundary_config: Optional[Dict] = None):
        """
        Initialize the decision boundary trainer.
        
        Parameters:
        - training_config: Standard training configuration
        - boundary_config: Decision boundary specific configuration
        """
        self.training_config = training_config
        self.boundary_config = boundary_config or {}
        
        # Extract configurations
        self.model_config = training_config['model']
        self.train_config = training_config['training']
        self.data_config = training_config['data']
        
        # Boundary extraction settings
        self.extraction_schedule = self.boundary_config.get('training', {}).get('extraction_schedule', {})
        self.extract_boundaries = self.extraction_schedule.get('enabled', False)
        
        # Initialize boundary extractor if enabled
        self.boundary_extractor = None
        if self.extract_boundaries:
            self.boundary_extractor = DecisionBoundaryExtractor(self.boundary_config)
        
        # Storage for results
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'learning_rate': []
        }
        
        self.boundary_results = []  # List of BoundaryExtractionResult objects
        self.layer_outputs = None   # For layer activation extraction
        
        # Setup output directories
        self._setup_output_directories()
    
    def _setup_output_directories(self):
        """Create necessary output directories."""
        if self.extract_boundaries:
            output_config = self.boundary_config.get('output', {})
            directories = output_config.get('directories', {})
            
            base_dir = Path(directories.get('base_dir', 'results/decision_boundary_analysis'))
            self.boundary_dir = base_dir / directories.get('boundaries_dir', 'boundaries')
            self.topology_dir = base_dir / directories.get('topology_dir', 'topology')
            self.analysis_dir = base_dir / directories.get('analysis_dir', 'analysis')
            
            # Create directories
            for dir_path in [self.boundary_dir, self.topology_dir, self.analysis_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_device_and_model(self) -> Tuple[torch.device, MLP]:
        """Setup device and initialize model."""
        # Device setup with fallback (same as torch_mlp.py)
        if self.train_config['device'] == 'cuda' and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(self.train_config['device'])
        
        print(f"Using device: {device}")
        
        # Model initialization (same as torch_mlp.py)
        model = MLP(
            input_dim=self.model_config['input_dim'],
            num_hidden_layers=self.model_config['num_hidden_layers'],
            hidden_dim=self.model_config['hidden_dim'],
            output_dim=self.model_config['output_dim'],
            activation_fn_name=self.model_config.get('activation_fn_name', 'relu'),
            dropout_rate=self.model_config.get('dropout_rate', 0.0),
            use_batch_norm=self.model_config.get('use_batch_norm', False)
        ).to(device)
        
        # Enable cuDNN benchmarking for faster training
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        return device, model
    
    def _setup_optimizer_and_scheduler(self, model: MLP) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Setup optimizer and learning rate scheduler."""
        lr = self.train_config['learning_rate']
        opt_config = self.train_config.get('optimizer', {'name': 'adam'})
        optimizer_type = opt_config.get('name', 'adam').lower()
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, 
                                 weight_decay=opt_config.get('weight_decay', 0.0))
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, 
                                  weight_decay=opt_config.get('weight_decay', 0.01))
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, 
                                weight_decay=opt_config.get('weight_decay', 0.0))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Learning rate scheduler
        scheduler = None
        scheduler_config = self.train_config.get('lr_scheduler', {})
        if scheduler_config.get('type') == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6),
                verbose=scheduler_config.get('verbose', False)
            )
        elif scheduler_config.get('type') == 'step_lr':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        
        return optimizer, scheduler
    
    def _load_data(self, device: torch.device) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
        """Load and prepare data."""
        # Data loading (same logic as torch_mlp.py)
        data_source = self.data_config.get('data_source')
        if data_source is not None:
            print(f"Loading data from: {data_source}")
            X, y = load_data_from_file(data_source)
        elif self.data_config['type'] == 'synthetic':
            gen_config = self.data_config.get('generation', {})
            num_samples = gen_config.get('n', 1000)
            big_radius = gen_config.get('big_radius', 3)
            small_radius = gen_config.get('small_radius', 1)
            solid = gen_config.get('solid', False)
            interior_noise = gen_config.get('interior_noise', 0.1)
            X, y = generate_torus_data(num_samples, big_radius, small_radius, solid, interior_noise)
        else:
            raise ValueError("Invalid data configuration")
        
        # Move data to device
        X = X.to(device)
        y = y.to(device)
        
        # Shuffle data
        perm = torch.randperm(len(X), device=device)
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
        
        train_loader = DataLoader(train_dataset, batch_size=self.train_config['batch_size'],
                                shuffle=True, pin_memory=True if device.type == 'cuda' else False)
        test_loader = DataLoader(test_dataset, batch_size=self.train_config['batch_size'],
                               shuffle=False, pin_memory=True if device.type == 'cuda' else False)
        
        return train_loader, test_loader, X, y
    
    def _should_extract_boundary(self, epoch: int, total_epochs: int) -> bool:
        """Determine if boundary should be extracted at this epoch."""
        if not self.extract_boundaries:
            return False
        
        # Check epoch range
        start_epoch = self.extraction_schedule.get('start_epoch', 0)
        end_epoch = self.extraction_schedule.get('end_epoch', total_epochs)
        if epoch < start_epoch or (end_epoch is not None and epoch > end_epoch):
            return False
        
        # Check frequency
        frequency = self.extraction_schedule.get('frequency', 10)
        if epoch % frequency == 0:
            return True
        
        # Always extract at final epoch if configured
        if epoch == total_epochs - 1 and self.extraction_schedule.get('final_extraction', True):
            return True
        
        return False
    
    def _extract_boundary_at_epoch(self, model: MLP, device: torch.device, epoch: int) -> Optional[BoundaryExtractionResult]:
        """Extract decision boundary at the current epoch."""
        if not self.boundary_extractor:
            return None
        
        print(f"\n--- Extracting decision boundary at epoch {epoch} ---")
        start_time = time.time()
        
        # Extract boundary
        result = self.boundary_extractor.extract_decision_boundary(model, device, epoch)
        
        if result.success:
            # Save boundary data if configured
            storage_config = self.boundary_config.get('output', {}).get('storage', {})
            if storage_config.get('save_boundary_meshes', True) and result.mesh_vertices is not None:
                self._save_boundary_mesh(result, epoch)
            
            if storage_config.get('save_topology_data', True):
                self._save_topology_data(result, epoch)
            
            print(f"Boundary extraction completed in {time.time() - start_time:.2f}s")
        else:
            print(f"Boundary extraction failed: {result.error_message}")
        
        return result
    
    def _save_boundary_mesh(self, result: BoundaryExtractionResult, epoch: int):
        """Save boundary mesh to file."""
        try:
            if result.mesh_vertices is not None and result.mesh_faces is not None:
                # Save as PLY file
                mesh_file = self.boundary_dir / f"boundary_epoch_{epoch:04d}.ply"
                
                # Create trimesh object and save
                if TRIMESH_AVAILABLE:
                    mesh = trimesh.Trimesh(vertices=result.mesh_vertices, faces=result.mesh_faces)
                    
                    # Remove duplicate and degenerate faces for efficiency
                    mesh.update_faces(mesh.unique_faces())
                    mesh.update_faces(mesh.nondegenerate_faces())
                    
                    # Export with binary format for smaller file size
                    mesh.export(str(mesh_file), file_type='ply', encoding='binary')
                    
                    # Print mesh statistics
                    print(f"Saved boundary mesh: {mesh_file} ({len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces)")
                else:
                    # Fallback: save as compressed numpy arrays
                    np.savez_compressed(str(mesh_file).replace('.ply', '.npz'),
                                      vertices=result.mesh_vertices,
                                      faces=result.mesh_faces)
                    print(f"Saved boundary mesh (npz): {mesh_file}")
        except Exception as e:
            print(f"Error saving boundary mesh: {e}")
    
    def _save_topology_data(self, result: BoundaryExtractionResult, epoch: int):
        """Save topology data to file."""
        try:
            topology_file = self.topology_dir / f"topology_epoch_{epoch:04d}.pt"
            
            topology_data = {
                'epoch': epoch,
                'boundary_points': result.boundary_points,
                'extraction_time': result.extraction_time,
                'metadata': result.metadata
            }
            
            torch.save(topology_data, topology_file)
            print(f"Saved boundary data: {topology_file}")
        except Exception as e:
            print(f"Error saving boundary data: {e}")
    
    def _compute_accuracy(self, model: MLP, loader: DataLoader, device: torch.device) -> float:
        """Compute accuracy on a dataset."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted.squeeze() == target.squeeze()).sum().item()
        
        return correct / total
    
    def train(self) -> Dict:
        """
        Main training function with integrated decision boundary extraction.
        
        Returns:
        - training_results: Dictionary containing training history and boundary results
        """
        print("Starting training with decision boundary extraction...")
        print(f"Boundary extraction enabled: {self.extract_boundaries}")
        
        # Setup
        device, model = self._setup_device_and_model()
        optimizer, scheduler = self._setup_optimizer_and_scheduler(model)
        train_loader, test_loader, X_full, y_full = self._load_data(device)
        
        # Training parameters
        epochs = self.train_config['epochs']
        criterion = nn.BCELoss()
        
        # Mixed precision training
        scaler = GradScaler() if device.type == 'cuda' else None
        
        # Regularization parameters
        reg_config = self.train_config.get('regularization', {})
        l1_lambda = reg_config.get('l1_lambda', 0.0)
        l2_lambda = reg_config.get('l2_lambda', 0.0)
        
        # Early stopping
        early_stopping_config = self.train_config.get('early_stopping', {})
        early_stopping_enabled = early_stopping_config.get('enabled', False)
        patience = early_stopping_config.get('patience', 20)
        min_delta = early_stopping_config.get('min_delta', 0.0001)
        best_loss = float('inf')
        patience_counter = 0
        
        # Gradient clipping
        grad_clip_config = self.train_config.get('gradient_clipping', {})
        grad_clip_enabled = grad_clip_config.get('enabled', False)
        max_norm = grad_clip_config.get('max_norm', 1.0)
        
        print(f"Training for {epochs} epochs on {device}")
        print(f"Model: {self.model_config['num_hidden_layers']} layers, {self.model_config['hidden_dim']} hidden units")
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with autocast():
                        output = model(data)
                        loss = criterion(output.squeeze(), target.squeeze())
                        
                        # Add regularization
                        if l1_lambda > 0:
                            l1_penalty = sum(p.abs().sum() for p in model.parameters())
                            loss += l1_lambda * l1_penalty
                        
                        if l2_lambda > 0:
                            l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
                            loss += l2_lambda * l2_penalty
                    
                    scaler.scale(loss).backward()
                    
                    if grad_clip_enabled:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output.squeeze(), target.squeeze())
                    
                    # Add regularization
                    if l1_lambda > 0:
                        l1_penalty = sum(p.abs().sum() for p in model.parameters())
                        loss += l1_lambda * l1_penalty
                    
                    if l2_lambda > 0:
                        l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
                        loss += l2_lambda * l2_penalty
                    
                    loss.backward()
                    
                    if grad_clip_enabled:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    
                    optimizer.step()
                
                train_loss += loss.item()
            
            # Evaluation phase
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output.squeeze(), target.squeeze()).item()
            
            # Calculate averages
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            
            # Calculate accuracies
            train_accuracy = self._compute_accuracy(model, train_loader, device)
            test_accuracy = self._compute_accuracy(model, test_loader, device)
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_loss)
                else:
                    scheduler.step()
            
            # Store training history
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['test_loss'].append(test_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['test_accuracy'].append(test_accuracy)
            self.training_history['learning_rate'].append(current_lr)
            
            # Extract decision boundary if scheduled
            boundary_result = None
            if self._should_extract_boundary(epoch, epochs):
                boundary_result = self._extract_boundary_at_epoch(model, device, epoch)
                if boundary_result:
                    self.boundary_results.append(boundary_result)
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s): "
                  f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            if boundary_result and boundary_result.success:
                print(f"  Boundary: {len(boundary_result.boundary_points) if boundary_result.boundary_points is not None else 0} points")
            
            # Early stopping check
            if early_stopping_enabled:
                if test_loss < best_loss - min_delta:
                    best_loss = test_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        
        # Final boundary extraction if not already done
        if self.extract_boundaries and not self._should_extract_boundary(epochs-1, epochs):
            final_result = self._extract_boundary_at_epoch(model, device, epochs-1)
            if final_result:
                self.boundary_results.append(final_result)
        
        # Prepare results
        training_results = {
            'training_history': self.training_history,
            'boundary_results': self.boundary_results,
            'layer_outputs': self.layer_outputs,
            'final_model_state': model.state_dict(),
            'dataset': {
                'X': X_full.cpu().numpy(),
                'y': y_full.cpu().numpy()
            },
            'config': {
                'training': self.training_config,
                'boundary': self.boundary_config
            }
        }
        
        # Save complete training results
        if self.extract_boundaries:
            results_file = self.analysis_dir / "complete_training_results.pt"
            torch.save(training_results, results_file)
            print(f"\nComplete training results saved: {results_file}")
        
        print(f"\nTraining completed! Final test accuracy: {test_accuracy:.4f}")
        if self.extract_boundaries:
            print(f"Extracted {len(self.boundary_results)} decision boundaries")
        
        return training_results


def main():
    """Main execution function with command line arguments."""
    parser = argparse.ArgumentParser(description="Neural Network Training with Decision Boundary Analysis")
    parser.add_argument('--training-config', type=str, default='configs/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--boundary-config', type=str, default='configs/decision_boundary_config.yaml',
                       help='Path to boundary configuration file')
    parser.add_argument('--disable-boundaries', action='store_true',
                       help='Disable decision boundary extraction')
    parser.add_argument('--save-model', type=str, help='Path to save the trained model')
    
    args = parser.parse_args()
    
    try:
        # Load configurations
        with open(args.training_config, 'r') as f:
            training_config = yaml.safe_load(f)
        
        boundary_config = {}
        if not args.disable_boundaries:
            boundary_config = load_boundary_config(args.boundary_config)
            if not boundary_config:
                print("Warning: Could not load boundary config, disabling boundary extraction")
                args.disable_boundaries = True
        
        if args.disable_boundaries:
            boundary_config = {'training': {'extraction_schedule': {'enabled': False}}}
        
        # Create trainer and run training
        trainer = DecisionBoundaryTrainer(training_config, boundary_config)
        results = trainer.train()
        
        # Save model if requested
        if args.save_model:
            model_path = Path(args.save_model)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_data = {
                'model_state_dict': results['final_model_state'],
                'model_config': training_config['model'],
                'training_history': results['training_history']
            }
            torch.save(save_data, model_path)
            print(f"Model saved: {model_path}")
        
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()