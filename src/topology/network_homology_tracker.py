"""
Network Homology Tracker Module

This module provides the main orchestration class for tracking the homological
evolution of neural networks during training. It integrates the graph construction,
simplicial complex creation, and persistence computation components.

Key Components:
- NetworkHomologyTracker: Main class that tracks homology during training
- HomologySnapshot: Data class for storing homology at a specific time
- NetworkHomologyHistory: Class for managing the history of homology computations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import time
from pathlib import Path
import pickle
import json
import yaml

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.network_graph_builder import UnifiedGraphBuilder
from src.utils.network_simplicial_complex import DirectedFlagComplex, compute_network_homology


@dataclass
class HomologySnapshot:
    """Data class for storing homology information at a specific training step."""
    step: int
    epoch: int
    batch_idx: int
    timestamp: float
    betti_numbers: np.ndarray
    persistence_diagrams: Dict[int, np.ndarray]
    distance_from_previous: Optional[float] = None
    validation_accuracy: Optional[float] = None
    train_loss: Optional[float] = None
    computation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
            "timestamp": self.timestamp,
            "betti_numbers": self.betti_numbers.tolist(),
            "persistence_diagrams": {
                str(k): v.tolist() for k, v in self.persistence_diagrams.items()
            },
            "distance_from_previous": self.distance_from_previous,
            "validation_accuracy": self.validation_accuracy,
            "train_loss": self.train_loss,
            "computation_time": self.computation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HomologySnapshot':
        """Create from dictionary."""
        return cls(
            step=data["step"],
            epoch=data["epoch"],
            batch_idx=data["batch_idx"],
            timestamp=data["timestamp"],
            betti_numbers=np.array(data["betti_numbers"]),
            persistence_diagrams={
                int(k): np.array(v) for k, v in data["persistence_diagrams"].items()
            },
            distance_from_previous=data.get("distance_from_previous"),
            validation_accuracy=data.get("validation_accuracy"),
            train_loss=data.get("train_loss"),
            computation_time=data.get("computation_time", 0.0)
        )


class NetworkHomologyHistory:
    """Manages the history of homology computations during training."""
    
    def __init__(self):
        """Initialize empty history."""
        self.snapshots: List[HomologySnapshot] = []
        self.metadata: Dict[str, Any] = {
            "start_time": time.time(),
            "model_architecture": None,
            "total_parameters": 0,
            "config": {}
        }
    
    def add_snapshot(self, snapshot: HomologySnapshot) -> None:
        """Add a new snapshot to the history."""
        self.snapshots.append(snapshot)
    
    def get_latest_snapshot(self) -> Optional[HomologySnapshot]:
        """Get the most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None
    
    def get_betti_evolution(self, dimension: int = 0) -> np.ndarray:
        """
        Get evolution of Betti numbers for a specific dimension.
        
        Args:
            dimension: Homological dimension
            
        Returns:
            Array of shape (num_snapshots,) with Betti numbers
        """
        betti_evolution = []
        for snapshot in self.snapshots:
            if dimension < len(snapshot.betti_numbers):
                betti_evolution.append(snapshot.betti_numbers[dimension])
            else:
                betti_evolution.append(0)
        return np.array(betti_evolution)
    
    def get_distance_evolution(self) -> np.ndarray:
        """Get evolution of distances between consecutive snapshots."""
        distances = []
        for snapshot in self.snapshots:
            if snapshot.distance_from_previous is not None:
                distances.append(snapshot.distance_from_previous)
        return np.array(distances)
    
    def get_validation_evolution(self) -> np.ndarray:
        """Get evolution of validation accuracy."""
        accuracies = []
        for snapshot in self.snapshots:
            if snapshot.validation_accuracy is not None:
                accuracies.append(snapshot.validation_accuracy)
        return np.array(accuracies)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save history to file."""
        filepath = Path(filepath)
        
        # Prepare data for serialization
        data = {
            "metadata": self.metadata,
            "snapshots": [s.to_dict() for s in self.snapshots]
        }
        
        if filepath.suffix == ".json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif filepath.suffix == ".pkl":
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            # Default to pickle
            with open(filepath.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(data, f)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load history from file."""
        filepath = Path(filepath)
        
        if filepath.suffix == ".json":
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        
        # Restore data
        self.metadata = data["metadata"]
        self.snapshots = [HomologySnapshot.from_dict(s) for s in data["snapshots"]]


class NetworkHomologyTracker:
    """
    Main class for tracking the homological evolution of neural networks.
    
    This class orchestrates the entire pipeline:
    1. Extracts network structure as a graph
    2. Constructs simplicial complex
    3. Computes persistent homology
    4. Tracks changes over training time
    5. Computes distances between consecutive states
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the homology tracker.
        
        Args:
            config: Configuration dictionary (or loads from default config)
        """
        if config is None:
            # Load default configuration
            config = self._load_default_config()
        
        self.config = config
        
        # Extract key parameters
        graph_config = config.get('graph_construction', {})
        self.normalize_weights = graph_config.get('normalize_weights', True)
        self.weight_threshold = graph_config.get('weight_threshold', 1e-6)
        self.handle_negative_weights = graph_config.get('handle_negative_weights', True)
        self.weight_encoding = graph_config.get('weight_encoding', 'standard')
        self.min_edge_distance = graph_config.get('min_edge_distance', 1e-6)
        self.normalization_type = graph_config.get('normalization_type', 'standard')
        
        complex_config = config.get('simplicial_complex', {})
        self.max_dimension = complex_config.get('max_dimension', 2)
        self.max_edge_length = complex_config.get('max_edge_length', 1.0)
        self.backend = complex_config.get('backend', 'auto')
        
        persistence_config = config.get('persistence', {})
        self.distance_metric = config.get('distance_metrics', {}).get('primary_metric', 'wasserstein')
        
        # Initialize components
        self.graph_builder = UnifiedGraphBuilder(
            normalize_weights=self.normalize_weights,
            weight_threshold=self.weight_threshold,
            handle_negative_weights=self.handle_negative_weights,
            weight_encoding=self.weight_encoding,
            min_edge_distance=self.min_edge_distance,
            normalization_type=self.normalization_type
        )
        
        self.complex_builder = DirectedFlagComplex(
            max_dimension=self.max_dimension,
            max_edge_length=self.max_edge_length,
            backend=self.backend
        )
        
        # Initialize history
        self.history = NetworkHomologyHistory()
        
        # Cache for graph structure (only weights change during training)
        self.cached_graph_structure = None
        self.cached_model_architecture = None
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from file."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "network_homology_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return minimal default config
            return {
                "graph_construction": {
                    "normalize_weights": True,
                    "weight_threshold": 1e-6,
                    "handle_negative_weights": True
                },
                "simplicial_complex": {
                    "max_dimension": 2,
                    "max_edge_length": 1.0,
                    "backend": "auto"
                },
                "distance_metrics": {
                    "primary_metric": "wasserstein"
                }
            }
    
    def track_training_step(self, model: nn.Module, 
                          step: int,
                          epoch: int = 0,
                          batch_idx: int = 0,
                          validation_accuracy: Optional[float] = None,
                          train_loss: Optional[float] = None) -> Tuple[float, HomologySnapshot]:
        """
        Track homology at a specific training step.
        
        Args:
            model: The neural network model
            step: Global training step
            epoch: Current epoch
            batch_idx: Batch index within epoch
            validation_accuracy: Optional validation accuracy
            train_loss: Optional training loss
            
        Returns:
            Tuple of (distance_from_previous, snapshot)
        """
        start_time = time.time()
        
        # Build network graph
        graph = self._build_network_graph(model)
        
        # Compute homology
        homology_result = compute_network_homology(
            graph,
            max_dimension=self.max_dimension,
            max_edge_length=self.max_edge_length,
            backend=self.backend
        )
        
        # Extract results
        betti_numbers = homology_result["betti_numbers"]
        persistence_diagrams = homology_result["persistence_diagrams"]
        
        # Compute distance from previous snapshot
        distance_from_previous = None
        previous_snapshot = self.history.get_latest_snapshot()
        
        if previous_snapshot is not None:
            distance_from_previous = self._compute_distance(
                persistence_diagrams,
                previous_snapshot.persistence_diagrams
            )
        
        # Create snapshot
        computation_time = time.time() - start_time
        snapshot = HomologySnapshot(
            step=step,
            epoch=epoch,
            batch_idx=batch_idx,
            timestamp=time.time(),
            betti_numbers=betti_numbers,
            persistence_diagrams=persistence_diagrams,
            distance_from_previous=distance_from_previous,
            validation_accuracy=validation_accuracy,
            train_loss=train_loss,
            computation_time=computation_time
        )
        
        # Add to history
        self.history.add_snapshot(snapshot)
        
        # Update metadata if first snapshot
        if len(self.history.snapshots) == 1:
            self.history.metadata["model_architecture"] = str(model)
            self.history.metadata["total_parameters"] = sum(
                p.numel() for p in model.parameters()
            )
            self.history.metadata["config"] = self.config
        
        return distance_from_previous or 0.0, snapshot
    
    def _build_network_graph(self, model: nn.Module):
        """Build or update network graph from model."""
        # Check if we can use cached structure
        model_str = str(model)
        if self.cached_model_architecture == model_str and self.cached_graph_structure is not None:
            # Only update weights in the cached graph
            return self._update_graph_weights(model)
        else:
            # Build new graph
            graph = self.graph_builder.build_network_graph(model)
            self.cached_graph_structure = graph
            self.cached_model_architecture = model_str
            return graph
    
    def _update_graph_weights(self, model: nn.Module):
        """Update weights in cached graph structure."""
        # This is an optimization - for now, just rebuild
        # In a production implementation, we would update only the edge weights
        return self.graph_builder.build_network_graph(model)
    
    def _compute_distance(self, diagrams1: Dict[int, np.ndarray], 
                         diagrams2: Dict[int, np.ndarray]) -> float:
        """
        Compute distance between two sets of persistence diagrams.
        
        Args:
            diagrams1: First set of persistence diagrams
            diagrams2: Second set of persistence diagrams
            
        Returns:
            Distance value
        """
        # Import distance computation module
        from src.analysis.persistence_distances import compute_all_distances
        
        # Get distance metric from config
        distance_config = self.config.get('distance_metrics', {})
        primary_metric = distance_config.get('primary_metric', 'heat')
        
        # Metric-specific parameters
        metric_params = {}
        if primary_metric == 'heat':
            metric_params['sigma'] = distance_config.get('heat_sigma', 0.1)
        elif primary_metric == 'wasserstein':
            metric_params['p'] = distance_config.get('wasserstein_p', 2)
        
        # Compute distance
        distances = compute_all_distances(
            diagrams1, diagrams2,
            metrics=[primary_metric],
            aggregate='sum',
            **metric_params
        )
        
        return distances[primary_metric]
    
    def compute_correlation_with_validation(self, window_size: Optional[int] = None) -> float:
        """
        Compute correlation between homology distance and validation accuracy.
        
        Args:
            window_size: Size of sliding window (None for all data)
            
        Returns:
            Correlation coefficient
        """
        distances = self.history.get_distance_evolution()
        validations = self.history.get_validation_evolution()
        
        if len(distances) < 2 or len(validations) < 2:
            return 0.0
        
        # Align arrays - ensure they have the same length
        min_len = min(len(distances), len(validations))
        if min_len < 2:
            return 0.0
            
        distances = distances[:min_len]
        validations = validations[:min_len]
        
        # Apply window if specified
        if window_size is not None and window_size < len(distances):
            distances = distances[-window_size:]
            validations = validations[-window_size:]
        
        # Compute correlation
        if len(distances) > 0 and len(validations) > 0:
            correlation = np.corrcoef(distances, validations)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save all results to output directory.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save history
        self.history.save(output_dir / "homology_history.pkl")
        
        # Save summary statistics
        summary = {
            "num_snapshots": len(self.history.snapshots),
            "total_computation_time": sum(s.computation_time for s in self.history.snapshots),
            "correlation_with_validation": self.compute_correlation_with_validation(),
            "final_betti_numbers": self.history.get_latest_snapshot().betti_numbers.tolist()
            if self.history.get_latest_snapshot() else None,
            "config": self.config
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save evolution data as CSV for easy analysis
        try:
            import pandas as pd
        except ImportError:
            print("Warning: pandas not available, skipping CSV export")
            return
        
        evolution_data = []
        for snapshot in self.history.snapshots:
            row = {
                "step": snapshot.step,
                "epoch": snapshot.epoch,
                "batch_idx": snapshot.batch_idx,
                "distance_from_previous": snapshot.distance_from_previous,
                "validation_accuracy": snapshot.validation_accuracy,
                "train_loss": snapshot.train_loss,
                "computation_time": snapshot.computation_time
            }
            
            # Add Betti numbers
            for i, b in enumerate(snapshot.betti_numbers):
                row[f"betti_{i}"] = b
            
            evolution_data.append(row)
        
        df = pd.DataFrame(evolution_data)
        df.to_csv(output_dir / "homology_evolution.csv", index=False)
        
        print(f"Results saved to {output_dir}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the homology tracking."""
        if not self.history.snapshots:
            return {"error": "No snapshots recorded"}
        
        distances = self.history.get_distance_evolution()
        
        return {
            "num_snapshots": len(self.history.snapshots),
            "total_computation_time": sum(s.computation_time for s in self.history.snapshots),
            "average_computation_time": np.mean([s.computation_time for s in self.history.snapshots]),
            "correlation_with_validation": self.compute_correlation_with_validation(),
            "distance_statistics": {
                "mean": float(np.mean(distances)) if len(distances) > 0 else 0.0,
                "std": float(np.std(distances)) if len(distances) > 0 else 0.0,
                "min": float(np.min(distances)) if len(distances) > 0 else 0.0,
                "max": float(np.max(distances)) if len(distances) > 0 else 0.0
            },
            "final_betti_numbers": self.history.get_latest_snapshot().betti_numbers.tolist()
            if self.history.get_latest_snapshot() else None
        }