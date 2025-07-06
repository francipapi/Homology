"""
Network Homology Comparison Script

This script compares the homological properties of multiple trained neural networks
by computing pairwise distances between their network graph persistence diagrams.

Key Features:
- Loads trained models from a specified directory
- Computes network homology using the graph framework
- Calculates pairwise distances using multiple metrics
- Generates visualizations including heatmaps and clustering
- Fully configurable via network_homology_config.yaml
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Import modules from the project
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.topology.network_homology_tracker import NetworkHomologyTracker
from src.analysis.persistence_distances import compute_all_distances, PersistenceDistanceCalculator
from src.models.torch_mlp import MLP
from src.models.torch_custom import CustomNet


class NetworkHomologyComparison:
    """
    Main class for comparing network homologies across multiple trained models.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the comparison tool.
        
        Args:
            config_path: Path to configuration file (uses default if None)
        """
        self.config = self._load_config(config_path)
        self.comparison_config = self.config['network_homology']['comparison']
        
        # Setup output directory
        self.output_dir = Path(self.comparison_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.matrices_dir = self.output_dir / "distance_matrices"
        self.viz_dir = self.output_dir / "visualizations"
        self.diagrams_dir = self.output_dir / "persistence_diagrams"
        
        for dir_path in [self.matrices_dir, self.viz_dir, self.diagrams_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize homology tracker
        self.homology_tracker = NetworkHomologyTracker(self.config)
        
        # Force use of GUDHI backend for persistence distance calculations since custom is buggy
        self.distance_calculator = PersistenceDistanceCalculator(backend="gudhi")
        
        # Storage for results
        self.model_paths: List[Path] = []
        self.model_metadata: Dict[str, Any] = {}
        self.persistence_diagrams: Dict[str, Dict[int, np.ndarray]] = {}
        self.distance_matrices: Dict[str, np.ndarray] = {}
        self.dimension_wise_distances: Dict[str, Dict[int, np.ndarray]] = {}  # Store per-dimension distances
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "network_homology_config.yaml"
        else:
            config_path = Path(config_path)
            
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_display_names(self, model_paths: List[Path]) -> List[str]:
        """
        Create clear display names for models in visualizations.
        
        Args:
            model_paths: List of model file paths
            
        Returns:
            List of display names for visualization
        """
        names = []
        for path in model_paths:
            name = path.stem  # Get filename without extension
            
            # Pattern-based name formatting
            if name.startswith('random_'):
                # random_mlp_net_000_default_seed_42 -> "Random MLP #0"
                # random_custom_net_000_default_seed_42 -> "Random Custom #0"
                parts = name.split('_')
                if 'mlp' in name:
                    arch = 'MLP'
                elif 'custom' in name:
                    arch = 'Custom'
                else:
                    arch = 'Net'
                
                # Extract number
                for part in parts:
                    if part.isdigit():
                        num = int(part)
                        break
                else:
                    num = 0
                
                names.append(f"Random {arch} #{num}")
                
            elif name.startswith('torch_'):
                # torch_mlp_acc_0.9857_epoch_100 -> "MLP (acc=98.6%)"
                # torch_custom_acc_1.0000_epoch_200 -> "Custom (acc=100%)"
                parts = name.split('_')
                
                # Extract architecture
                if 'mlp' in name:
                    arch = 'MLP'
                elif 'custom' in name:
                    arch = 'Custom'
                else:
                    arch = 'Model'
                
                # Extract accuracy
                acc_str = ""
                for i, part in enumerate(parts):
                    if part == 'acc' and i + 1 < len(parts):
                        try:
                            acc = float(parts[i + 1])
                            acc_str = f" (acc={acc*100:.1f}%)"
                        except ValueError:
                            pass
                        break
                
                # Extract epoch
                epoch_str = ""
                for i, part in enumerate(parts):
                    if part == 'epoch' and i + 1 < len(parts):
                        try:
                            epoch = int(parts[i + 1])
                            epoch_str = f" @{epoch}ep"
                        except ValueError:
                            pass
                        break
                
                names.append(f"{arch}{acc_str}{epoch_str}")
                
            else:
                # Fallback: try to create readable name
                # Replace underscores with spaces and title case
                readable = name.replace('_', ' ').title()
                # Limit length
                if len(readable) > 20:
                    readable = readable[:17] + "..."
                names.append(readable)
        
        # Check for duplicates and add suffixes if needed
        seen = {}
        final_names = []
        for name in names:
            if name in seen:
                seen[name] += 1
                final_names.append(f"{name} ({seen[name]})")
            else:
                seen[name] = 0
                final_names.append(name)
        
        return final_names

    def _infer_mlp_config(self, state_dict: Dict[str, torch.Tensor], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer MLP configuration from state_dict structure.
        
        Args:
            state_dict: Model state dictionary
            model_config: Config from saved model (may be incomplete)
            
        Returns:
            Complete configuration dictionary
        """
        # Find layer weight tensors
        layer_weights = {}
        for key, tensor in state_dict.items():
            if 'layers.' in key and '.weight' in key:
                layer_idx = int(key.split('.')[1])
                layer_weights[layer_idx] = tensor
        
        if not layer_weights:
            # Fallback to config defaults
            return {
                'input_dim': model_config.get('input_dim', 3),
                'num_hidden_layers': model_config.get('num_hidden_layers', 2),
                'hidden_dim': model_config.get('hidden_dim', 64),
                'output_dim': model_config.get('output_dim', 1),
                'activation_fn_name': model_config.get('activation_fn_name', 'relu'),
                'dropout_rate': model_config.get('dropout_rate', 0.0),
                'use_batch_norm': model_config.get('use_batch_norm', False)
            }
        
        # Infer dimensions from weight shapes
        sorted_layers = sorted(layer_weights.keys())
        first_layer = layer_weights[sorted_layers[0]]
        last_layer = layer_weights[sorted_layers[-1]]
        
        input_dim = first_layer.shape[1]  # Input features
        output_dim = last_layer.shape[0]  # Output features
        
        # Infer hidden dimensions (assume all hidden layers have same size)
        if len(sorted_layers) > 1:
            hidden_dim = first_layer.shape[0]  # Hidden layer size
            num_hidden_layers = len(sorted_layers) - 1  # All layers except output
        else:
            hidden_dim = 64  # Default
            num_hidden_layers = 0
        
        return {
            'input_dim': input_dim,
            'num_hidden_layers': num_hidden_layers,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'activation_fn_name': model_config.get('activation_fn_name', 'relu'),
            'dropout_rate': model_config.get('dropout_rate', 0.0),
            'use_batch_norm': model_config.get('use_batch_norm', False)
        }

    def find_model_files(self) -> List[Path]:
        """Find all model files based on configuration."""
        models_dir = Path(self.comparison_config['models_dir'])
        if not models_dir.exists():
            raise ValueError(f"Models directory not found: {models_dir}")
        
        # Find all matching model files
        pattern = self.comparison_config['model_filter']
        if '*' in pattern:
            model_files = list(models_dir.rglob(pattern))
        else:
            model_files = list(models_dir.rglob(f"**/{pattern}"))
        
        # Sort for consistent ordering
        model_files.sort()
        
        # Limit number of models if specified
        max_models = self.comparison_config['max_models']
        if max_models is not None and len(model_files) > max_models:
            print(f"Limiting to {max_models} models (found {len(model_files)})")
            model_files = model_files[:max_models]
        
        if not model_files:
            raise ValueError(f"No model files found matching pattern '{pattern}' in {models_dir}")
        
        print(f"Found {len(model_files)} model files")
        return model_files
    
    def load_model_safely(self, model_path: Path) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load a model from file with error handling.
        
        Returns:
            Tuple of (model, metadata)
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract config and metadata - handle different config key formats
            config = checkpoint.get('config', {})
            model_config = None
            custom_config = None
            
            # Try different config key formats
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
            elif 'config' in checkpoint and 'model' in checkpoint['config']:
                model_config = checkpoint['config']['model']
            
            if 'custom_config' in checkpoint:
                custom_config = checkpoint['custom_config']
            elif 'config' in checkpoint and 'custom_architecture' in checkpoint['config']:
                custom_config = checkpoint['config']['custom_architecture']
            
            metadata = {
                'path': str(model_path),
                'parent_dir': model_path.parent.name,
                'best_validation_accuracy': checkpoint.get('best_validation_accuracy', None),
                'final_accuracy': checkpoint.get('final_accuracy', None),
                'config': config,
                'model_config': model_config,
                'custom_config': custom_config
            }
            
            # Determine model type by checking the state dict structure
            state_dict = checkpoint['model_state_dict']
            
            # Check if it's a custom architecture based on layer names
            layer_keys = [k for k in state_dict.keys() if 'layers.' in k]
            if layer_keys:
                # Extract layer indices to determine architecture
                layer_indices = sorted(set(int(k.split('.')[1]) for k in layer_keys if k.split('.')[1].isdigit()))
                
                # Check if this looks like a custom architecture (non-sequential layer indices)
                is_custom = False
                if len(layer_indices) > 0:
                    # If indices are not sequential or have gaps, it's likely custom
                    expected_indices = list(range(min(layer_indices), max(layer_indices) + 1))
                    if layer_indices != expected_indices:
                        is_custom = True
                
                if is_custom or (custom_config is not None and custom_config.get('enabled', False)):
                    # Try to infer custom architecture or use config
                    if custom_config is not None:
                        model = CustomNet(custom_config)
                    else:
                        # Fall back to standard MLP if we can't determine custom config
                        inferred_config = self._infer_mlp_config(state_dict, model_config or {})
                        model = MLP(
                            input_dim=inferred_config['input_dim'],
                            num_hidden_layers=inferred_config['num_hidden_layers'],
                            hidden_dim=inferred_config['hidden_dim'],
                            output_dim=inferred_config['output_dim'],
                            activation_fn_name=inferred_config['activation_fn_name'],
                            dropout_rate=inferred_config['dropout_rate'],
                            use_batch_norm=inferred_config['use_batch_norm']
                        )
                else:
                    # Standard MLP - infer architecture from state_dict
                    inferred_config = self._infer_mlp_config(state_dict, model_config or {})
                    
                    model = MLP(
                        input_dim=inferred_config['input_dim'],
                        num_hidden_layers=inferred_config['num_hidden_layers'],
                        hidden_dim=inferred_config['hidden_dim'],
                        output_dim=inferred_config['output_dim'],
                        activation_fn_name=inferred_config['activation_fn_name'],
                        dropout_rate=inferred_config['dropout_rate'],
                        use_batch_norm=inferred_config['use_batch_norm']
                    )
            else:
                # Default to MLP if no clear layer structure
                inferred_config = self._infer_mlp_config(state_dict, model_config or {})
                
                model = MLP(
                    input_dim=inferred_config['input_dim'],
                    num_hidden_layers=inferred_config['num_hidden_layers'],
                    hidden_dim=inferred_config['hidden_dim'],
                    output_dim=inferred_config['output_dim'],
                    activation_fn_name=inferred_config['activation_fn_name'],
                    dropout_rate=inferred_config['dropout_rate'],
                    use_batch_norm=inferred_config['use_batch_norm']
                )
            
            # Try to load model weights
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                # If loading fails, try to load with strict=False to handle architecture mismatches
                print(f"Warning: Architecture mismatch for {model_path}, attempting flexible loading...")
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            model.eval()
            
            # Add input_shape attribute required by UnifiedGraphBuilder
            if not hasattr(model, 'input_shape'):
                if hasattr(model, 'input_dim'):
                    model.input_shape = (model.input_dim,)
                else:
                    # Fallback to config
                    input_dim = config.get('model', {}).get('input_dim', 3)
                    model.input_shape = (input_dim,)
            
            return model, metadata
            
        except Exception as e:
            print(f"Error loading model {model_path}: {str(e)}")
            raise
    
    def compute_model_homology(self, model: nn.Module, model_name: str) -> Dict[int, np.ndarray]:
        """
        Compute persistent homology for a single model.
        
        Args:
            model: The neural network model
            model_name: Identifier for the model
            
        Returns:
            Persistence diagrams indexed by dimension
        """
        # Use homology tracker to compute homology
        # We pass dummy training parameters since we're only interested in the homology
        distance, snapshot = self.homology_tracker.track_training_step(
            model=model,
            step=0,
            epoch=0,
            batch_idx=0
        )
        
        return snapshot.persistence_diagrams
    
    def compute_all_homologies(self) -> None:
        """Compute homology for all models."""
        print("\nComputing network homology for all models...")
        
        for i, model_path in enumerate(tqdm(self.model_paths, desc="Computing homologies")):
            try:
                # Load model
                model, metadata = self.load_model_safely(model_path)
                
                # Store metadata
                model_name = f"model_{i:03d}_{model_path.parent.name}"
                self.model_metadata[model_name] = metadata
                
                # Compute homology
                persistence_diagrams = self.compute_model_homology(model, model_name)
                self.persistence_diagrams[model_name] = persistence_diagrams
                
                # Optionally save individual persistence diagrams
                if self.comparison_config['output']['save_persistence_diagrams']:
                    diagram_path = self.diagrams_dir / f"{model_name}_persistence.pkl"
                    with open(diagram_path, 'wb') as f:
                        pickle.dump(persistence_diagrams, f)
                
                # Clear model from memory
                del model
                
            except Exception as e:
                print(f"\nError processing {model_path}: {str(e)}")
                continue
    
    def compute_pairwise_distances(self) -> None:
        """Compute pairwise distances between all models."""
        print("\nComputing pairwise distances...")
        
        model_names = list(self.persistence_diagrams.keys())
        n_models = len(model_names)
        
        # Get metrics to compute
        metrics = self.comparison_config['metrics']
        metric_params = self.comparison_config.get('metric_params', {})
        
        # Initialize distance matrices
        for metric in metrics:
            self.distance_matrices[metric] = np.zeros((n_models, n_models))
            self.dimension_wise_distances[metric] = {}
        
        # Compute pairwise distances
        total_pairs = n_models * (n_models - 1) // 2
        with tqdm(total=total_pairs, desc="Computing distances") as pbar:
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    # Get persistence diagrams
                    diagrams1 = self.persistence_diagrams[model_names[i]]
                    diagrams2 = self.persistence_diagrams[model_names[j]]
                    
                    # Compute distances using GUDHI backend
                    distances = {}
                    for metric in metrics:
                        # Get metric-specific parameters
                        if metric == 'heat':
                            kwargs = {'sigma': metric_params.get('heat_sigma', 0.1)}
                        elif metric == 'wasserstein':
                            kwargs = {
                                'p': metric_params.get('wasserstein_p', 2.0),
                                'delta': metric_params.get('wasserstein_delta', 0.01)
                            }
                        else:
                            kwargs = {}
                        
                        # Get all dimensions that exist in either diagram
                        all_dims = set(diagrams1.keys()) | set(diagrams2.keys())
                        
                        # Determine maximum dimension to consider
                        max_dim_config = self.config.get('network_homology', {}).get('max_dimension', None)
                        if max_dim_config is not None:
                            # Filter dimensions based on config
                            all_dims = {dim for dim in all_dims if dim <= max_dim_config}
                        
                        if not all_dims:
                            # If no dimensions available, distance is 0
                            distances[metric] = 0.0
                            continue
                        
                        # Compute distance for each dimension
                        dimension_distances = {}
                        total_distance = 0.0
                        
                        for dim in sorted(all_dims):
                            dgm1 = diagrams1.get(dim, np.empty((0, 2)))
                            dgm2 = diagrams2.get(dim, np.empty((0, 2)))
                            
                            # Ensure diagrams have correct shape
                            if dgm1.size == 0:
                                dgm1 = np.empty((0, 2))
                            if dgm2.size == 0:
                                dgm2 = np.empty((0, 2))
                            
                            # Compute distance for this dimension
                            try:
                                dim_dist = self.distance_calculator.compute_distance(
                                    dgm1, dgm2, metric=metric, **kwargs
                                )
                                dimension_distances[dim] = dim_dist
                                
                                # Store dimension-wise distance matrix
                                if dim not in self.dimension_wise_distances[metric]:
                                    self.dimension_wise_distances[metric][dim] = np.zeros((n_models, n_models))
                                
                                # Weight distances by dimension (optional)
                                weight = metric_params.get(f'dim_{dim}_weight', 1.0)
                                total_distance += weight * dim_dist
                                
                            except Exception as e:
                                print(f"Warning: Error computing {metric} distance for dimension {dim}: {e}")
                                dimension_distances[dim] = 0.0
                        
                        # Apply aggregation method
                        aggregation_method = metric_params.get('aggregation_method', 'sum')
                        if aggregation_method == 'sum':
                            distances[metric] = total_distance
                        elif aggregation_method == 'max':
                            distances[metric] = max(dimension_distances.values()) if dimension_distances else 0.0
                        elif aggregation_method == 'mean':
                            distances[metric] = np.mean(list(dimension_distances.values())) if dimension_distances else 0.0
                        elif aggregation_method == 'weighted_sum':
                            # Already computed above with weights
                            distances[metric] = total_distance
                        else:
                            # Default to sum
                            distances[metric] = total_distance
                    
                    # Store in matrices (symmetric)
                    for metric in metrics:
                        dist = distances[metric]
                        self.distance_matrices[metric][i, j] = dist
                        self.distance_matrices[metric][j, i] = dist
                        
                        # Store dimension-wise distances
                        for dim, dim_dist in dimension_distances.items():
                            if dim in self.dimension_wise_distances[metric]:
                                self.dimension_wise_distances[metric][dim][i, j] = dim_dist
                                self.dimension_wise_distances[metric][dim][j, i] = dim_dist
                    
                    pbar.update(1)
        
        print(f"Computed {len(metrics)} distance metrics for {n_models} models")
        
        # Print dimension information
        all_dims = set()
        for diagrams in self.persistence_diagrams.values():
            all_dims.update(diagrams.keys())
        if all_dims:
            print(f"Computed distances for homology dimensions: {sorted(all_dims)}")
    
    def save_distance_matrices(self) -> None:
        """Save computed distance matrices."""
        if not self.comparison_config['output']['save_distance_matrices']:
            return
        
        print("\nSaving distance matrices...")
        format_type = self.comparison_config['output']['distance_matrix_format']
        save_dimension_wise = self.comparison_config['output'].get('save_dimension_wise_matrices', True)
        
        # Save overall distance matrices
        for metric, matrix in self.distance_matrices.items():
            if format_type == 'npy':
                np.save(self.matrices_dir / f"{metric}_distance_matrix.npy", matrix)
            elif format_type == 'npz':
                save_data = {
                    'distance_matrix': matrix,
                    'model_names': list(self.persistence_diagrams.keys())
                }
                # Add dimension-wise matrices if available
                if save_dimension_wise and metric in self.dimension_wise_distances:
                    for dim, dim_matrix in self.dimension_wise_distances[metric].items():
                        save_data[f'dim_{dim}_matrix'] = dim_matrix
                
                np.savez_compressed(
                    self.matrices_dir / f"{metric}_distance_matrix.npz",
                    **save_data
                )
            elif format_type == 'csv':
                # Save as CSV with model names
                import pandas as pd
                model_names = list(self.persistence_diagrams.keys())
                df = pd.DataFrame(matrix, index=model_names, columns=model_names)
                df.to_csv(self.matrices_dir / f"{metric}_distance_matrix.csv")
                
                # Save dimension-wise matrices
                if save_dimension_wise and metric in self.dimension_wise_distances:
                    for dim, dim_matrix in self.dimension_wise_distances[metric].items():
                        df_dim = pd.DataFrame(dim_matrix, index=model_names, columns=model_names)
                        df_dim.to_csv(self.matrices_dir / f"{metric}_distance_matrix_dim_{dim}.csv")
                        
            elif format_type == 'pickle':
                save_data = {
                    'matrix': matrix,
                    'model_names': list(self.persistence_diagrams.keys())
                }
                # Add dimension-wise matrices
                if save_dimension_wise and metric in self.dimension_wise_distances:
                    save_data['dimension_wise_matrices'] = self.dimension_wise_distances[metric]
                
                with open(self.matrices_dir / f"{metric}_distance_matrix.pkl", 'wb') as f:
                    pickle.dump(save_data, f)
        
        # Save dimension-wise summary
        if save_dimension_wise:
            dims_summary = {}
            for metric in self.dimension_wise_distances:
                dims_summary[metric] = list(self.dimension_wise_distances[metric].keys())
            
            with open(self.matrices_dir / "dimension_summary.json", 'w') as f:
                json.dump({
                    'dimensions_computed': dims_summary,
                    'total_models': len(self.persistence_diagrams),
                    'metrics': list(self.distance_matrices.keys())
                }, f, indent=2)
    
    def create_visualizations(self) -> None:
        """Create heatmap visualizations and clustering analysis."""
        if not self.comparison_config['visualization']['create_heatmap']:
            return
        
        print("\nCreating visualizations...")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        viz_config = self.comparison_config['visualization']
        
        # Create clear display names for all models
        display_names = self._create_display_names(self.model_paths)
        
        for metric, matrix in self.distance_matrices.items():
            # Create figure
            fig, ax = plt.subplots(figsize=viz_config['figsize'])
            
            # Use improved model names
            
            # Normalize matrix for better visualization
            if matrix.max() > 0:
                normalized_matrix = matrix / matrix.max()
            else:
                normalized_matrix = matrix
            
            # Create heatmap
            sns.heatmap(
                normalized_matrix,
                annot=viz_config['annotate_heatmap'],
                fmt=viz_config['annotation_format'],
                cmap=viz_config['heatmap_cmap'],
                square=True,
                cbar_kws={'label': f'Normalized {metric.capitalize()} Distance'},
                ax=ax
            )
            
            # Set labels with improved names
            ax.set_xticklabels(display_names, rotation=45, ha='right')
            ax.set_yticklabels(display_names, rotation=0)
            ax.set_title(f'Network Homology Distance Matrix ({metric.capitalize()})')
            
            # Save figure
            plt.tight_layout()
            output_path = self.viz_dir / f"{metric}_heatmap.{viz_config['heatmap_format']}"
            plt.savefig(
                output_path,
                dpi=viz_config['heatmap_dpi'],
                bbox_inches='tight'
            )
            plt.close()
        
        # Perform clustering if enabled
        if self.comparison_config['clustering']['perform_clustering']:
            self._perform_clustering_analysis()
    
    def _perform_clustering_analysis(self) -> None:
        """Perform clustering analysis on distance matrices."""
        clustering_config = self.comparison_config['clustering']
        
        # Use the first metric's distance matrix for clustering
        primary_metric = self.comparison_config['metrics'][0]
        distance_matrix = self.distance_matrices[primary_metric]
        
        if clustering_config['method'] == 'hierarchical':
            self._hierarchical_clustering(distance_matrix, primary_metric)
        elif clustering_config['method'] == 'spectral':
            self._spectral_clustering(distance_matrix, primary_metric)
        elif clustering_config['method'] == 'dbscan':
            self._dbscan_clustering(distance_matrix, primary_metric)
    
    def _hierarchical_clustering(self, distance_matrix: np.ndarray, metric_name: str) -> None:
        """Perform hierarchical clustering and create dendrogram."""
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        
        clustering_config = self.comparison_config['clustering']
        
        # Convert distance matrix to condensed form
        condensed_distances = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method=clustering_config['linkage'])
        
        if clustering_config['create_dendrogram']:
            # Create dendrogram
            fig, ax = plt.subplots(figsize=(12, 8))
            
            display_names = self._create_display_names(self.model_paths)
            
            dendrogram(
                linkage_matrix,
                labels=display_names,
                ax=ax,
                orientation='top',
                distance_sort='descending'
            )
            
            ax.set_title(f'Hierarchical Clustering Dendrogram ({metric_name.capitalize()} Distance)')
            ax.set_xlabel('Model')
            ax.set_ylabel('Distance')
            
            plt.tight_layout()
            output_path = self.viz_dir / f"clustering_dendrogram_{metric_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _spectral_clustering(self, distance_matrix: np.ndarray, metric_name: str) -> None:
        """Perform spectral clustering."""
        from sklearn.cluster import SpectralClustering
        
        clustering_config = self.comparison_config['clustering']
        
        # Convert distance matrix to affinity matrix
        # Using Gaussian kernel: exp(-distance^2 / (2 * median^2))
        median_dist = np.median(distance_matrix[distance_matrix > 0])
        affinity_matrix = np.exp(-distance_matrix**2 / (2 * median_dist**2))
        
        # Determine number of clusters
        n_clusters = clustering_config['num_clusters']
        if n_clusters is None:
            # Simple elbow method using eigenvalues
            eigenvalues = np.linalg.eigvals(affinity_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            # Find elbow (simplified)
            n_clusters = min(5, len(self.persistence_diagrams) // 2)
        
        # Perform clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        labels = clustering.fit_predict(affinity_matrix)
        
        # Save clustering results
        clustering_results = {
            'method': 'spectral',
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'model_names': list(self.persistence_diagrams.keys())
        }
        
        with open(self.output_dir / f"clustering_results_{metric_name}.json", 'w') as f:
            json.dump(clustering_results, f, indent=2)
    
    def save_metadata(self) -> None:
        """Save model metadata and comparison information."""
        if not self.comparison_config['output']['save_metadata']:
            return
        
        print("\nSaving metadata...")
        
        metadata = {
            'comparison_info': {
                'n_models': len(self.model_paths),
                'metrics_computed': list(self.distance_matrices.keys()),
                'computation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': self.comparison_config
            },
            'model_info': self.model_metadata
        }
        
        format_type = self.comparison_config['output']['metadata_format']
        
        if format_type == 'json':
            with open(self.output_dir / 'model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        elif format_type == 'yaml':
            with open(self.output_dir / 'model_metadata.yaml', 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
        elif format_type == 'pickle':
            with open(self.output_dir / 'model_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
    
    def create_summary_report(self) -> None:
        """Create a summary report of the comparison results."""
        if not self.comparison_config['output']['create_summary_report']:
            return
        
        print("\nCreating summary report...")
        
        # Collect dimension information
        all_dims = set()
        for diagrams in self.persistence_diagrams.values():
            all_dims.update(diagrams.keys())
        
        summary = {
            'overview': {
                'total_models': len(self.model_paths),
                'models_processed': len(self.persistence_diagrams),
                'metrics_computed': list(self.distance_matrices.keys()),
                'homology_dimensions': sorted(all_dims) if all_dims else [],
                'max_dimension': max(all_dims) if all_dims else 0
            },
            'distance_statistics': {},
            'dimension_wise_statistics': {}
        }
        
        # Compute statistics for each metric
        for metric, matrix in self.distance_matrices.items():
            # Get upper triangular values (excluding diagonal)
            upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
            
            summary['distance_statistics'][metric] = {
                'aggregated': {
                    'mean': float(np.mean(upper_tri)),
                    'std': float(np.std(upper_tri)),
                    'min': float(np.min(upper_tri)),
                    'max': float(np.max(upper_tri)),
                    'median': float(np.median(upper_tri))
                }
            }
            
            # Add dimension-wise statistics
            if metric in self.dimension_wise_distances:
                summary['dimension_wise_statistics'][metric] = {}
                for dim, dim_matrix in self.dimension_wise_distances[metric].items():
                    dim_upper_tri = dim_matrix[np.triu_indices_from(dim_matrix, k=1)]
                    summary['dimension_wise_statistics'][metric][f'dimension_{dim}'] = {
                        'mean': float(np.mean(dim_upper_tri)),
                        'std': float(np.std(dim_upper_tri)),
                        'min': float(np.min(dim_upper_tri)),
                        'max': float(np.max(dim_upper_tri)),
                        'median': float(np.median(dim_upper_tri))
                    }
        
        # Add model grouping statistics if available
        model_groups = {}
        for name in self.persistence_diagrams.keys():
            group = name.split('_')[-1]  # Extract model type from name
            if group not in model_groups:
                model_groups[group] = []
            model_groups[group].append(name)
        
        summary['model_groups'] = {k: len(v) for k, v in model_groups.items()}
        
        # Save summary
        with open(self.output_dir / 'comparison_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print dimension analysis to console
        self._print_dimension_analysis(summary)
    
    def _print_dimension_analysis(self, summary: Dict[str, Any]) -> None:
        """Print dimension-wise analysis to console."""
        print("\n" + "=" * 60)
        print("HOMOLOGY DIMENSION ANALYSIS")
        print("=" * 60)
        
        dims = summary['overview']['homology_dimensions']
        if not dims:
            print("No homology dimensions found.")
            return
        
        # Print model information
        print(f"Total models compared: {len(self.model_paths)}")
        print(f"Models processed successfully: {len(self.persistence_diagrams)}")
        
        print("\nModels analyzed:")
        display_names = self._create_display_names(self.model_paths)
        for i, (path, display_name) in enumerate(zip(self.model_paths, display_names)):
            model_name = f"model_{i:03d}_{path.parent.name}"
            
            # Get model metadata if available
            metadata = self.model_metadata.get(model_name, {})
            accuracy_info = ""
            if metadata.get('best_validation_accuracy') is not None:
                accuracy_info = f" (acc={metadata['best_validation_accuracy']:.3f})"
            elif metadata.get('final_accuracy') is not None:
                accuracy_info = f" (acc={metadata['final_accuracy']:.3f})"
            
            print(f"  {i+1:2d}. {display_name}{accuracy_info}")
            print(f"      Path: {path.name}")
        
        print(f"\nHomology dimensions computed: {dims}")
        print(f"Maximum dimension: {summary['overview']['max_dimension']}")
        
        # Print pairwise distance summary
        self._print_pairwise_summary(display_names)
        
        # Print dimension-wise statistics
        self._print_dimension_analysis_stats(summary)
        
    def _print_pairwise_summary(self, display_names: List[str]) -> None:
        """Print a summary of pairwise distances between models."""
        if not self.distance_matrices:
            return
            
        print("\n" + "-" * 60)
        print("PAIRWISE DISTANCE SUMMARY")
        print("-" * 60)
        
        # Use the first metric for summary
        first_metric = list(self.distance_matrices.keys())[0]
        distance_matrix = self.distance_matrices[first_metric]
        
        # Find most/least similar pairs
        n_models = len(display_names)
        if n_models < 2:
            return
            
        # Get upper triangle indices (avoid diagonal and duplicates)
        triu_indices = np.triu_indices(n_models, k=1)
        distances = distance_matrix[triu_indices]
        
        if len(distances) == 0:
            return
            
        # Find extremes
        min_idx = np.argmin(distances)
        max_idx = np.argmax(distances)
        
        min_i, min_j = triu_indices[0][min_idx], triu_indices[1][min_idx]
        max_i, max_j = triu_indices[0][max_idx], triu_indices[1][max_idx]
        
        print(f"Most similar models ({first_metric} distance):")
        print(f"  {display_names[min_i]} ↔ {display_names[min_j]}")
        print(f"  Distance: {distances[min_idx]:.4f}")
        
        print(f"\nMost dissimilar models ({first_metric} distance):")
        print(f"  {display_names[max_i]} ↔ {display_names[max_j]}")
        print(f"  Distance: {distances[max_idx]:.4f}")
        
        print(f"\nOverall distance statistics ({first_metric}):")
        print(f"  Mean: {np.mean(distances):.4f}")
        print(f"  Std:  {np.std(distances):.4f}")
        print(f"  Range: [{np.min(distances):.4f}, {np.max(distances):.4f}]")

    def _print_dimension_analysis_stats(self, summary: Dict[str, Any]) -> None:
        """Print dimension-wise statistics."""
        # Print dimension-wise statistics
        if 'dimension_wise_statistics' in summary:
            for metric, dim_stats in summary['dimension_wise_statistics'].items():
                print(f"\n{metric.upper()} DISTANCE BY DIMENSION:")
                print("-" * 40)
                
                for dim_key, stats in dim_stats.items():
                    dim_num = dim_key.replace('dimension_', '')
                    print(f"  Dimension {dim_num}:")
                    print(f"    Mean: {stats['mean']:.6f}")
                    print(f"    Std:  {stats['std']:.6f}")
                    print(f"    Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
                
                # Compare with aggregated
                agg_stats = summary['distance_statistics'][metric]['aggregated']
                print(f"  Aggregated (all dimensions):")
                print(f"    Mean: {agg_stats['mean']:.6f}")
                print(f"    Std:  {agg_stats['std']:.6f}")
                print(f"    Range: [{agg_stats['min']:.6f}, {agg_stats['max']:.6f}]")
        
        print("=" * 60)
    
    def run(self) -> None:
        """Run the complete comparison pipeline."""
        print("=" * 60)
        print("Network Homology Comparison Tool")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Find model files
            self.model_paths = self.find_model_files()
            
            # Compute homology for all models
            self.compute_all_homologies()
            
            if len(self.persistence_diagrams) < 2:
                print("Error: Need at least 2 models for comparison")
                return
            
            # Compute pairwise distances
            self.compute_pairwise_distances()
            
            # Save results
            self.save_distance_matrices()
            self.save_metadata()
            
            # Create visualizations
            self.create_visualizations()
            
            # Create summary report
            self.create_summary_report()
            
            elapsed_time = time.time() - start_time
            print(f"\nComparison completed in {elapsed_time:.2f} seconds")
            print(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"\nError during comparison: {str(e)}")
            raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Compare network homology across multiple trained models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: configs/network_homology_config.yaml)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Override models directory from config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Override maximum number of models to compare"
    )
    
    args = parser.parse_args()
    
    # Create comparison tool
    comparison = NetworkHomologyComparison(args.config)
    
    # Override config if command-line arguments provided
    if args.models_dir:
        comparison.comparison_config['models_dir'] = args.models_dir
    if args.output_dir:
        comparison.comparison_config['output_dir'] = args.output_dir
        comparison.output_dir = Path(args.output_dir)
        comparison.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_models is not None:
        comparison.comparison_config['max_models'] = args.max_models
    
    # Run comparison
    comparison.run()


if __name__ == "__main__":
    main()