"""
Gradient-Based and Optimization Landscape Similarity Measures

This module implements various gradient-based similarity measures for neural networks
that capture functional behavior through optimization dynamics, loss landscapes,
and gradient flow topology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import eigvalsh
from scipy.stats import wasserstein_distance
import warnings
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class GradientFlowSnapshot:
    """Container for gradient flow information at a specific point."""
    parameters: torch.Tensor
    gradients: torch.Tensor
    loss: float
    step: int
    lr: float
    velocity: Optional[torch.Tensor] = None  # For momentum-based optimizers
    
    
@dataclass
class LossLandscapePoint:
    """Container for loss landscape analysis at a point."""
    parameters: torch.Tensor
    loss: float
    gradient: torch.Tensor
    hessian: Optional[torch.Tensor] = None
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = None
    

class GradientSimilarityAnalyzer:
    """
    Comprehensive analyzer for gradient-based similarity measures between neural networks.
    
    Implements:
    1. Gradient Flow Topology analysis
    2. Loss Landscape analysis
    3. Hessian-based similarity
    4. Neural Tangent Kernel (NTK) analysis
    5. Optimization dynamics tracking
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize the analyzer with specified device."""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_flow_history = defaultdict(list)
        self.loss_landscape_cache = {}
        
    # ============ 1. Gradient Flow Topology ============
    
    def track_gradient_flow(self, model: nn.Module, loss_fn: Callable, 
                           data_loader: torch.utils.data.DataLoader,
                           optimizer: torch.optim.Optimizer,
                           num_steps: int = 100) -> List[GradientFlowSnapshot]:
        """
        Track the gradient flow trajectory during optimization.
        
        Args:
            model: Neural network model
            loss_fn: Loss function
            data_loader: Data loader for training data
            optimizer: Optimizer instance
            num_steps: Number of optimization steps to track
            
        Returns:
            List of gradient flow snapshots
        """
        snapshots = []
        model.train()
        
        for step in range(num_steps):
            total_loss = 0
            total_grad = []
            
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                
                # Collect gradients before optimizer step
                grads = []
                params = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads.append(param.grad.clone().detach().flatten())
                        params.append(param.clone().detach().flatten())
                
                total_loss += loss.item()
                
                if batch_idx == 0:  # Track full gradient for first batch
                    total_grad = torch.cat(grads) if grads else torch.tensor([])
                
                optimizer.step()
                
                if batch_idx >= 5:  # Limit to few batches per step for efficiency
                    break
            
            # Create snapshot
            all_params = torch.cat(params) if params else torch.tensor([])
            snapshot = GradientFlowSnapshot(
                parameters=all_params,
                gradients=total_grad,
                loss=total_loss / (batch_idx + 1),
                step=step,
                lr=optimizer.param_groups[0]['lr']
            )
            snapshots.append(snapshot)
            
        return snapshots
    
    def compute_gradient_flow_similarity(self, flow1: List[GradientFlowSnapshot], 
                                       flow2: List[GradientFlowSnapshot],
                                       method: str = 'trajectory') -> float:
        """
        Compute similarity between two gradient flow trajectories.
        
        Args:
            flow1, flow2: Gradient flow trajectories
            method: Similarity method ('trajectory', 'velocity', 'curvature')
            
        Returns:
            Similarity score
        """
        if method == 'trajectory':
            # Compute Fréchet distance between trajectories
            return self._frechet_distance(
                [s.parameters.cpu().numpy() for s in flow1],
                [s.parameters.cpu().numpy() for s in flow2]
            )
        
        elif method == 'velocity':
            # Compare gradient magnitudes and directions
            velocities1 = [s.gradients.norm().item() for s in flow1]
            velocities2 = [s.gradients.norm().item() for s in flow2]
            return 1.0 - wasserstein_distance(velocities1, velocities2) / max(max(velocities1), max(velocities2))
        
        elif method == 'curvature':
            # Analyze trajectory curvature
            curvatures1 = self._compute_trajectory_curvature(flow1)
            curvatures2 = self._compute_trajectory_curvature(flow2)
            return np.corrcoef(curvatures1, curvatures2)[0, 1]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def analyze_critical_points(self, model: nn.Module, loss_fn: Callable,
                              data_loader: torch.utils.data.DataLoader,
                              num_perturbations: int = 50) -> Dict[str, Any]:
        """
        Analyze critical points in the optimization landscape using Morse theory.
        
        Args:
            model: Neural network model
            loss_fn: Loss function
            data_loader: Data loader
            num_perturbations: Number of random perturbations to test
            
        Returns:
            Dictionary with critical point analysis
        """
        model.eval()
        original_params = self._get_flat_params(model)
        
        critical_points = {
            'minima': [],
            'saddles': [],
            'maxima': [],
            'morse_index': []
        }
        
        # Test perturbations around current point
        for _ in range(num_perturbations):
            # Random perturbation
            perturbation = torch.randn_like(original_params) * 0.01
            self._set_flat_params(model, original_params + perturbation)
            
            # Compute Hessian eigenvalues
            eigenvalues = self._compute_hessian_eigenvalues(model, loss_fn, data_loader, top_k=20)
            
            # Classify critical point by eigenvalues
            num_negative = sum(eigenvalues < -1e-6)
            num_positive = sum(eigenvalues > 1e-6)
            
            if num_negative == 0:
                critical_points['minima'].append(eigenvalues)
            elif num_positive == 0:
                critical_points['maxima'].append(eigenvalues)
            else:
                critical_points['saddles'].append(eigenvalues)
            
            critical_points['morse_index'].append(num_negative)
        
        # Restore original parameters
        self._set_flat_params(model, original_params)
        
        return {
            'num_minima': len(critical_points['minima']),
            'num_saddles': len(critical_points['saddles']),
            'num_maxima': len(critical_points['maxima']),
            'average_morse_index': np.mean(critical_points['morse_index']),
            'morse_index_distribution': np.histogram(critical_points['morse_index'], bins=10)[0]
        }
    
    # ============ 2. Loss Landscape Analysis ============
    
    def analyze_loss_landscape(self, model: nn.Module, loss_fn: Callable,
                             data_loader: torch.utils.data.DataLoader,
                             directions: Optional[List[torch.Tensor]] = None,
                             resolution: int = 50, epsilon: float = 0.1) -> Dict[str, Any]:
        """
        Analyze the loss landscape around the current parameters.
        
        Args:
            model: Neural network model
            loss_fn: Loss function
            data_loader: Data loader
            directions: Principal directions to analyze (if None, uses random)
            resolution: Number of points to sample in each direction
            epsilon: Range of perturbation
            
        Returns:
            Loss landscape analysis results
        """
        model.eval()
        original_params = self._get_flat_params(model)
        
        if directions is None:
            # Use random orthogonal directions
            d = len(original_params)
            directions = [torch.randn(d).to(self.device) for _ in range(2)]
            # Orthogonalize
            directions[1] = directions[1] - (directions[1] @ directions[0]) * directions[0]
            directions = [d / d.norm() for d in directions]
        
        # Create grid
        alphas = torch.linspace(-epsilon, epsilon, resolution)
        loss_surface = torch.zeros(resolution, resolution)
        
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(alphas):
                # Perturb parameters
                perturbed = original_params + alpha * directions[0] + beta * directions[1]
                self._set_flat_params(model, perturbed)
                
                # Compute loss
                total_loss = 0
                with torch.no_grad():
                    for data, target in data_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        total_loss += loss_fn(output, target).item()
                
                loss_surface[i, j] = total_loss / len(data_loader)
        
        # Restore parameters
        self._set_flat_params(model, original_params)
        
        # Analyze landscape properties
        return {
            'loss_surface': loss_surface.numpy(),
            'roughness': self._compute_landscape_roughness(loss_surface),
            'convexity': self._compute_landscape_convexity(loss_surface),
            'barrier_heights': self._compute_barrier_heights(loss_surface),
            'basin_volume': self._estimate_basin_volume(loss_surface)
        }
    
    def compare_loss_landscapes(self, landscape1: Dict[str, Any], 
                               landscape2: Dict[str, Any]) -> Dict[str, float]:
        """
        Compare two loss landscapes using various metrics.
        
        Args:
            landscape1, landscape2: Loss landscape analysis results
            
        Returns:
            Dictionary of similarity metrics
        """
        surface1 = landscape1['loss_surface']
        surface2 = landscape2['loss_surface']
        
        # Normalize surfaces for comparison
        surface1_norm = (surface1 - surface1.mean()) / surface1.std()
        surface2_norm = (surface2 - surface2.mean()) / surface2.std()
        
        return {
            'surface_correlation': np.corrcoef(surface1_norm.flatten(), 
                                              surface2_norm.flatten())[0, 1],
            'roughness_ratio': landscape1['roughness'] / (landscape2['roughness'] + 1e-8),
            'convexity_difference': abs(landscape1['convexity'] - landscape2['convexity']),
            'basin_volume_ratio': landscape1['basin_volume'] / (landscape2['basin_volume'] + 1e-8),
            'barrier_correlation': np.corrcoef(landscape1['barrier_heights'], 
                                             landscape2['barrier_heights'])[0, 1]
        }
    
    # ============ 3. Hessian-Based Similarity ============
    
    def compute_hessian_similarity(self, model1: nn.Module, model2: nn.Module,
                                 loss_fn: Callable, data_loader: torch.utils.data.DataLoader,
                                 method: str = 'eigenvalue', top_k: int = 50) -> float:
        """
        Compute similarity between models using Hessian information.
        
        Args:
            model1, model2: Neural network models
            loss_fn: Loss function
            data_loader: Data loader
            method: Similarity method ('eigenvalue', 'trace', 'determinant', 'condition')
            top_k: Number of top eigenvalues to consider
            
        Returns:
            Similarity score
        """
        # Compute top eigenvalues for both models
        eigenvalues1 = self._compute_hessian_eigenvalues(model1, loss_fn, data_loader, top_k)
        eigenvalues2 = self._compute_hessian_eigenvalues(model2, loss_fn, data_loader, top_k)
        
        if method == 'eigenvalue':
            # Compare eigenvalue distributions
            return 1.0 - wasserstein_distance(eigenvalues1, eigenvalues2) / (
                max(eigenvalues1.max(), eigenvalues2.max()) + 1e-8
            )
        
        elif method == 'trace':
            # Compare traces (sum of eigenvalues)
            trace1 = eigenvalues1.sum()
            trace2 = eigenvalues2.sum()
            return 1.0 - abs(trace1 - trace2) / (abs(trace1) + abs(trace2) + 1e-8)
        
        elif method == 'determinant':
            # Compare log determinants
            log_det1 = np.log(np.abs(eigenvalues1) + 1e-8).sum()
            log_det2 = np.log(np.abs(eigenvalues2) + 1e-8).sum()
            return 1.0 - abs(log_det1 - log_det2) / (abs(log_det1) + abs(log_det2) + 1e-8)
        
        elif method == 'condition':
            # Compare condition numbers
            cond1 = eigenvalues1.max() / (eigenvalues1.min() + 1e-8)
            cond2 = eigenvalues2.max() / (eigenvalues2.min() + 1e-8)
            return 1.0 / (1.0 + abs(np.log(cond1) - np.log(cond2)))
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def analyze_curvature(self, model: nn.Module, loss_fn: Callable,
                         data_loader: torch.utils.data.DataLoader,
                         num_directions: int = 20) -> Dict[str, Any]:
        """
        Analyze local curvature properties using second-order information.
        
        Args:
            model: Neural network model
            loss_fn: Loss function
            data_loader: Data loader
            num_directions: Number of random directions to analyze
            
        Returns:
            Curvature analysis results
        """
        model.eval()
        original_params = self._get_flat_params(model)
        d = len(original_params)
        
        directional_curvatures = []
        
        for _ in range(num_directions):
            # Random direction
            direction = torch.randn(d).to(self.device)
            direction = direction / direction.norm()
            
            # Compute directional second derivative
            curvature = self._compute_directional_curvature(
                model, loss_fn, data_loader, direction
            )
            directional_curvatures.append(curvature)
        
        directional_curvatures = np.array(directional_curvatures)
        
        return {
            'mean_curvature': directional_curvatures.mean(),
            'max_curvature': directional_curvatures.max(),
            'min_curvature': directional_curvatures.min(),
            'curvature_variance': directional_curvatures.var(),
            'negative_curvature_ratio': (directional_curvatures < 0).mean(),
            'curvature_distribution': directional_curvatures
        }
    
    # ============ 4. Neural Tangent Kernel (NTK) ============
    
    def compute_ntk(self, model: nn.Module, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute the Neural Tangent Kernel between two inputs.
        
        Args:
            model: Neural network model
            x1, x2: Input tensors
            
        Returns:
            NTK matrix
        """
        model.eval()
        x1, x2 = x1.to(self.device), x2.to(self.device)
        
        # Compute outputs
        y1 = model(x1)
        y2 = model(x2)
        
        # Initialize kernel
        K = torch.zeros(x1.size(0), x2.size(0)).to(self.device)
        
        # Compute kernel by summing over output dimensions
        for i in range(y1.size(1)):
            # Compute gradients for each output dimension
            grad1 = torch.autograd.grad(y1[:, i].sum(), model.parameters(), 
                                       retain_graph=True, create_graph=True)
            grad2 = torch.autograd.grad(y2[:, i].sum(), model.parameters(), 
                                       retain_graph=True, create_graph=True)
            
            # Compute inner product of gradients
            for g1, g2 in zip(grad1, grad2):
                if g1 is not None and g2 is not None:
                    K += (g1.flatten().unsqueeze(0) @ g2.flatten().unsqueeze(1).T)
        
        return K
    
    def compare_ntk_similarity(self, model1: nn.Module, model2: nn.Module,
                             data_loader: torch.utils.data.DataLoader,
                             num_samples: int = 100) -> Dict[str, float]:
        """
        Compare two models using their Neural Tangent Kernels.
        
        Args:
            model1, model2: Neural network models
            data_loader: Data loader
            num_samples: Number of samples to use
            
        Returns:
            Dictionary of NTK similarity metrics
        """
        # Get sample data
        data_iter = iter(data_loader)
        x_samples = []
        for _ in range(min(num_samples // data_loader.batch_size + 1, len(data_loader))):
            try:
                batch = next(data_iter)
                x_samples.append(batch[0])
            except StopIteration:
                break
        
        x_samples = torch.cat(x_samples)[:num_samples].to(self.device)
        
        # Compute NTK matrices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore gradient warnings
            K1 = self.compute_ntk(model1, x_samples, x_samples)
            K2 = self.compute_ntk(model2, x_samples, x_samples)
        
        # Normalize kernels
        K1_norm = K1 / (K1.norm() + 1e-8)
        K2_norm = K2 / (K2.norm() + 1e-8)
        
        # Compute similarity metrics
        return {
            'kernel_alignment': (K1_norm * K2_norm).sum().item(),
            'frobenius_similarity': 1.0 - (K1_norm - K2_norm).norm().item() / 2.0,
            'eigenvalue_similarity': self._compare_kernel_eigenvalues(K1, K2),
            'trace_similarity': 1.0 - abs(K1.trace() - K2.trace()) / (
                abs(K1.trace()) + abs(K2.trace()) + 1e-8
            )
        }
    
    # ============ 5. Optimization Dynamics ============
    
    def track_optimization_dynamics(self, model: nn.Module, loss_fn: Callable,
                                  data_loader: torch.utils.data.DataLoader,
                                  optimizer_class: type, lr: float,
                                  num_epochs: int = 10) -> Dict[str, List[float]]:
        """
        Track various optimization dynamics metrics during training.
        
        Args:
            model: Neural network model
            loss_fn: Loss function
            data_loader: Data loader
            optimizer_class: Optimizer class (e.g., torch.optim.Adam)
            lr: Learning rate
            num_epochs: Number of epochs to track
            
        Returns:
            Dictionary of tracked metrics
        """
        optimizer = optimizer_class(model.parameters(), lr=lr)
        model.train()
        
        metrics = {
            'loss': [],
            'gradient_norm': [],
            'parameter_change': [],
            'effective_lr': [],
            'gradient_variance': [],
            'gradient_cosine': []
        }
        
        prev_params = self._get_flat_params(model).clone()
        prev_grad = None
        
        for epoch in range(num_epochs):
            epoch_metrics = defaultdict(list)
            
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                
                # Track gradient statistics
                current_grad = torch.cat([p.grad.flatten() for p in model.parameters() 
                                         if p.grad is not None])
                grad_norm = current_grad.norm().item()
                
                epoch_metrics['loss'].append(loss.item())
                epoch_metrics['gradient_norm'].append(grad_norm)
                
                # Gradient variance across parameters
                grad_var = current_grad.var().item()
                epoch_metrics['gradient_variance'].append(grad_var)
                
                # Gradient alignment with previous step
                if prev_grad is not None:
                    cosine_sim = F.cosine_similarity(current_grad.unsqueeze(0), 
                                                    prev_grad.unsqueeze(0)).item()
                    epoch_metrics['gradient_cosine'].append(cosine_sim)
                
                prev_grad = current_grad.clone()
                
                optimizer.step()
                
                # Track parameter change
                current_params = self._get_flat_params(model)
                param_change = (current_params - prev_params).norm().item()
                epoch_metrics['parameter_change'].append(param_change)
                
                # Effective learning rate
                if grad_norm > 0:
                    effective_lr = param_change / grad_norm
                    epoch_metrics['effective_lr'].append(effective_lr)
                
                prev_params = current_params.clone()
            
            # Aggregate epoch metrics
            for key, values in epoch_metrics.items():
                if values:
                    metrics[key].append(np.mean(values))
        
        return metrics
    
    def compare_optimization_dynamics(self, dynamics1: Dict[str, List[float]],
                                    dynamics2: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compare optimization dynamics between two training runs.
        
        Args:
            dynamics1, dynamics2: Optimization dynamics from track_optimization_dynamics
            
        Returns:
            Dictionary of similarity metrics
        """
        similarities = {}
        
        for key in dynamics1.keys():
            if key in dynamics2:
                seq1 = np.array(dynamics1[key])
                seq2 = np.array(dynamics2[key])
                
                # Ensure same length
                min_len = min(len(seq1), len(seq2))
                seq1 = seq1[:min_len]
                seq2 = seq2[:min_len]
                
                # Compute various similarity metrics
                similarities[f'{key}_correlation'] = np.corrcoef(seq1, seq2)[0, 1]
                similarities[f'{key}_rmse'] = np.sqrt(np.mean((seq1 - seq2)**2))
                similarities[f'{key}_dtw'] = self._dynamic_time_warping(seq1, seq2)
        
        # Overall dynamics similarity
        correlations = [v for k, v in similarities.items() if 'correlation' in k]
        similarities['overall_dynamics_similarity'] = np.mean(correlations)
        
        return similarities
    
    # ============ Helper Methods ============
    
    def _get_flat_params(self, model: nn.Module) -> torch.Tensor:
        """Get flattened model parameters."""
        return torch.cat([p.flatten() for p in model.parameters()])
    
    def _set_flat_params(self, model: nn.Module, params: torch.Tensor):
        """Set model parameters from flattened tensor."""
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data = params[offset:offset+numel].view(p.shape)
            offset += numel
    
    def _compute_hessian_eigenvalues(self, model: nn.Module, loss_fn: Callable,
                                   data_loader: torch.utils.data.DataLoader,
                                   top_k: int = 50) -> np.ndarray:
        """
        Compute top-k eigenvalues of the Hessian using power iteration.
        
        This is an approximation for efficiency.
        """
        model.eval()
        
        # Use a subset of data for efficiency
        data_subset = []
        target_subset = []
        for i, (data, target) in enumerate(data_loader):
            if i >= 5:  # Use only first 5 batches
                break
            data_subset.append(data)
            target_subset.append(target)
        
        data_subset = torch.cat(data_subset).to(self.device)
        target_subset = torch.cat(target_subset).to(self.device)
        
        def loss_func():
            output = model(data_subset)
            return loss_fn(output, target_subset)
        
        # Power iteration for top eigenvalues
        eigenvalues = []
        d = sum(p.numel() for p in model.parameters())
        
        for _ in range(min(top_k, d)):
            # Random initialization
            v = torch.randn(d).to(self.device)
            v = v / v.norm()
            
            # Power iteration
            for _ in range(20):  # Fixed iterations
                # Compute Hv using finite differences
                epsilon = 1e-3
                params = self._get_flat_params(model)
                
                # Forward difference
                self._set_flat_params(model, params + epsilon * v)
                loss_plus = loss_func()
                grad_plus = torch.autograd.grad(loss_plus, model.parameters(), create_graph=True)
                grad_plus_flat = torch.cat([g.flatten() for g in grad_plus])
                
                # Backward difference
                self._set_flat_params(model, params - epsilon * v)
                loss_minus = loss_func()
                grad_minus = torch.autograd.grad(loss_minus, model.parameters(), create_graph=True)
                grad_minus_flat = torch.cat([g.flatten() for g in grad_minus])
                
                # Restore parameters
                self._set_flat_params(model, params)
                
                # Hessian-vector product
                Hv = (grad_plus_flat - grad_minus_flat) / (2 * epsilon)
                
                # Update eigenvector
                eigenvalue = (v @ Hv).item()
                v = Hv / (Hv.norm() + 1e-8)
            
            eigenvalues.append(eigenvalue)
        
        return np.array(eigenvalues)
    
    def _compute_directional_curvature(self, model: nn.Module, loss_fn: Callable,
                                     data_loader: torch.utils.data.DataLoader,
                                     direction: torch.Tensor) -> float:
        """Compute curvature in a specific direction."""
        epsilon = 1e-3
        params = self._get_flat_params(model)
        
        # Compute loss at three points
        losses = []
        for delta in [-epsilon, 0, epsilon]:
            self._set_flat_params(model, params + delta * direction)
            
            total_loss = 0
            with torch.no_grad():
                for i, (data, target) in enumerate(data_loader):
                    if i >= 5:  # Use subset for efficiency
                        break
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    total_loss += loss_fn(output, target).item()
            
            losses.append(total_loss)
        
        # Restore parameters
        self._set_flat_params(model, params)
        
        # Finite difference approximation of second derivative
        curvature = (losses[2] - 2*losses[1] + losses[0]) / (epsilon**2)
        return curvature
    
    def _frechet_distance(self, curve1: List[np.ndarray], 
                         curve2: List[np.ndarray]) -> float:
        """Compute discrete Fréchet distance between curves."""
        n, m = len(curve1), len(curve2)
        ca = np.array(curve1)
        cb = np.array(curve2)
        
        # Dynamic programming table
        dp = np.full((n, m), -1.0)
        
        def _c(i, j):
            """Distance between points."""
            return np.linalg.norm(ca[i] - cb[j])
        
        def _compute(i, j):
            """Recursive computation with memoization."""
            if dp[i, j] > -1:
                return dp[i, j]
            
            if i == 0 and j == 0:
                dp[i, j] = _c(0, 0)
            elif i > 0 and j == 0:
                dp[i, j] = max(_compute(i-1, 0), _c(i, 0))
            elif i == 0 and j > 0:
                dp[i, j] = max(_compute(0, j-1), _c(0, j))
            else:
                dp[i, j] = max(min(_compute(i-1, j), _compute(i-1, j-1), 
                                  _compute(i, j-1)), _c(i, j))
            
            return dp[i, j]
        
        return _compute(n-1, m-1)
    
    def _compute_trajectory_curvature(self, flow: List[GradientFlowSnapshot]) -> np.ndarray:
        """Compute curvature along a gradient flow trajectory."""
        if len(flow) < 3:
            return np.array([0.0])
        
        positions = [s.parameters.cpu().numpy() for s in flow]
        curvatures = []
        
        for i in range(1, len(positions) - 1):
            # Three consecutive points
            p0, p1, p2 = positions[i-1], positions[i], positions[i+1]
            
            # Vectors
            v1 = p1 - p0
            v2 = p2 - p1
            
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            # Approximate curvature
            distance = np.linalg.norm(v1) + np.linalg.norm(v2)
            curvature = 2 * angle / distance if distance > 0 else 0
            curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def _compute_landscape_roughness(self, surface: torch.Tensor) -> float:
        """Compute roughness of loss surface."""
        # Total variation
        dx = torch.diff(surface, dim=0)
        dy = torch.diff(surface, dim=1)
        roughness = torch.abs(dx).mean() + torch.abs(dy).mean()
        return roughness.item()
    
    def _compute_landscape_convexity(self, surface: torch.Tensor) -> float:
        """Estimate convexity of loss surface."""
        # Check if surface is approximately convex
        n = surface.shape[0]
        center = n // 2
        center_value = surface[center, center]
        
        # Check convexity along multiple rays
        convexity_violations = 0
        total_checks = 0
        
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            for r in range(1, n//2):
                i = int(center + r * np.cos(angle))
                j = int(center + r * np.sin(angle))
                
                if 0 <= i < n and 0 <= j < n:
                    # Linear interpolation
                    expected = center_value + (surface[i, j] - center_value) * (r / (n//2))
                    
                    # Check intermediate points
                    for r_mid in range(1, r):
                        i_mid = int(center + r_mid * np.cos(angle))
                        j_mid = int(center + r_mid * np.sin(angle))
                        
                        if 0 <= i_mid < n and 0 <= j_mid < n:
                            actual = surface[i_mid, j_mid]
                            expected_mid = center_value + (expected - center_value) * (r_mid / r)
                            
                            if actual > expected_mid:
                                convexity_violations += 1
                            total_checks += 1
        
        return 1.0 - (convexity_violations / max(total_checks, 1))
    
    def _compute_barrier_heights(self, surface: torch.Tensor) -> np.ndarray:
        """Compute barrier heights in loss surface."""
        # Find local minima
        from scipy.ndimage import minimum_filter
        local_min = (surface == minimum_filter(surface.numpy(), size=3))
        min_indices = np.where(local_min)
        
        if len(min_indices[0]) < 2:
            return np.array([0.0])
        
        barriers = []
        
        # For each pair of minima, find the lowest barrier
        for i in range(len(min_indices[0])):
            for j in range(i+1, len(min_indices[0])):
                # Simple approximation: maximum along straight line
                start = (min_indices[0][i], min_indices[1][i])
                end = (min_indices[0][j], min_indices[1][j])
                
                # Sample points along line
                num_samples = 20
                line_values = []
                
                for t in np.linspace(0, 1, num_samples):
                    x = int(start[0] + t * (end[0] - start[0]))
                    y = int(start[1] + t * (end[1] - start[1]))
                    
                    if 0 <= x < surface.shape[0] and 0 <= y < surface.shape[1]:
                        line_values.append(surface[x, y].item())
                
                if line_values:
                    barrier = max(line_values) - min(surface[start].item(), 
                                                   surface[end].item())
                    barriers.append(barrier)
        
        return np.array(barriers) if barriers else np.array([0.0])
    
    def _estimate_basin_volume(self, surface: torch.Tensor) -> float:
        """Estimate volume of loss basin."""
        # Find minimum
        min_val = surface.min().item()
        min_idx = torch.where(surface == surface.min())
        center = (min_idx[0][0].item(), min_idx[1][0].item())
        
        # Estimate basin by flood fill up to threshold
        threshold = min_val + 0.1 * (surface.mean() - min_val).item()
        basin_mask = surface < threshold
        
        # Connected component around minimum
        from scipy.ndimage import label
        labeled, _ = label(basin_mask.numpy())
        basin_label = labeled[center]
        basin_size = (labeled == basin_label).sum()
        
        # Normalize by total size
        return basin_size / surface.numel()
    
    def _compare_kernel_eigenvalues(self, K1: torch.Tensor, K2: torch.Tensor) -> float:
        """Compare eigenvalue distributions of two kernels."""
        # Compute top eigenvalues
        try:
            eigvals1 = torch.linalg.eigvalsh(K1)[-50:].cpu().numpy()
            eigvals2 = torch.linalg.eigvalsh(K2)[-50:].cpu().numpy()
            
            # Normalize
            eigvals1 = eigvals1 / (eigvals1.sum() + 1e-8)
            eigvals2 = eigvals2 / (eigvals2.sum() + 1e-8)
            
            # Compare distributions
            return 1.0 - wasserstein_distance(eigvals1, eigvals2)
        except:
            return 0.0
    
    def _dynamic_time_warping(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute DTW distance between sequences."""
        n, m = len(seq1), len(seq2)
        dtw = np.full((n+1, m+1), np.inf)
        dtw[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        
        # Normalize by path length
        return dtw[n, m] / (n + m)
    
    
    def visualize_gradient_flow(self, flow: List[GradientFlowSnapshot], 
                               save_path: Optional[str] = None):
        """Visualize gradient flow trajectory."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss trajectory
        losses = [s.loss for s in flow]
        steps = [s.step for s in flow]
        axes[0, 0].plot(steps, losses)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Trajectory')
        
        # Gradient norm
        grad_norms = [s.gradients.norm().item() for s in flow]
        axes[0, 1].plot(steps, grad_norms)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].set_title('Gradient Magnitude')
        axes[0, 1].set_yscale('log')
        
        # Parameter change
        param_changes = []
        for i in range(1, len(flow)):
            change = (flow[i].parameters - flow[i-1].parameters).norm().item()
            param_changes.append(change)
        axes[1, 0].plot(steps[1:], param_changes)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Parameter Change')
        axes[1, 0].set_title('Step Size')
        
        # Learning rate
        lrs = [s.lr for s in flow]
        axes[1, 1].plot(steps, lrs)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_loss_landscape(self, landscape: Dict[str, Any], 
                                save_path: Optional[str] = None):
        """Visualize loss landscape analysis."""
        fig = plt.figure(figsize=(15, 5))
        
        # 3D surface plot
        ax1 = fig.add_subplot(131, projection='3d')
        surface = landscape['loss_surface']
        x = np.arange(surface.shape[0])
        y = np.arange(surface.shape[1])
        X, Y = np.meshgrid(x, y)
        ax1.plot_surface(X, Y, surface.T, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Direction 1')
        ax1.set_ylabel('Direction 2')
        ax1.set_zlabel('Loss')
        ax1.set_title('Loss Surface')
        
        # Contour plot
        ax2 = fig.add_subplot(132)
        contour = ax2.contour(surface, levels=20, cmap='viridis')
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.set_xlabel('Direction 1')
        ax2.set_ylabel('Direction 2')
        ax2.set_title('Loss Contours')
        
        # Metrics
        ax3 = fig.add_subplot(133)
        metrics = {
            'Roughness': landscape['roughness'],
            'Convexity': landscape['convexity'],
            'Basin Volume': landscape['basin_volume'],
            'Mean Barrier': np.mean(landscape['barrier_heights'])
        }
        
        y_pos = np.arange(len(metrics))
        ax3.barh(y_pos, list(metrics.values()))
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(list(metrics.keys()))
        ax3.set_xlabel('Value')
        ax3.set_title('Landscape Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()