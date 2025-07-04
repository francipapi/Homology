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
            
            # Extract config and metadata
            config = checkpoint.get('config', {})
            metadata = {
                'path': str(model_path),
                'parent_dir': model_path.parent.name,
                'best_validation_accuracy': checkpoint.get('best_validation_accuracy', None),
                'config': config
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
                
                if is_custom or ('custom_architecture' in config and config.get('custom_architecture', {}).get('enabled', False)):
                    # Try to infer custom architecture or use config
                    if 'custom_architecture' in config:
                        model = CustomNet(config['custom_architecture'])
                    else:
                        # Fall back to standard MLP if we can't determine custom config
                        model_config = config.get('model', {})
                        model = MLP(
                            input_dim=model_config.get('input_dim', 3),
                            num_hidden_layers=model_config.get('num_hidden_layers', 2),
                            hidden_dim=model_config.get('hidden_dim', 64),
                            output_dim=model_config.get('output_dim', 1),
                            activation_fn_name=model_config.get('activation_fn_name', 'relu'),
                            dropout_rate=model_config.get('dropout_rate', 0.0),
                            use_batch_norm=model_config.get('use_batch_norm', False)
                        )
                else:
                    # Standard MLP
                    model_config = config.get('model', {})
                    model = MLP(
                        input_dim=model_config.get('input_dim', 3),
                        num_hidden_layers=model_config.get('num_hidden_layers', 2),
                        hidden_dim=model_config.get('hidden_dim', 64),
                        output_dim=model_config.get('output_dim', 1),
                        activation_fn_name=model_config.get('activation_fn_name', 'relu'),
                        dropout_rate=model_config.get('dropout_rate', 0.0),
                        use_batch_norm=model_config.get('use_batch_norm', False)
                    )
            else:
                # Default to MLP if no clear layer structure
                model_config = config.get('model', {})
                model = MLP(
                    input_dim=model_config.get('input_dim', 3),
                    num_hidden_layers=model_config.get('num_hidden_layers', 2),
                    hidden_dim=model_config.get('hidden_dim', 64),
                    output_dim=model_config.get('output_dim', 1),
                    activation_fn_name=model_config.get('activation_fn_name', 'relu'),
                    dropout_rate=model_config.get('dropout_rate', 0.0),
                    use_batch_norm=model_config.get('use_batch_norm', False)
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
                        total_distance = 0.0
                        
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
                        
                        # Compute distance for each dimension and sum
                        max_dim = max(max(diagrams1.keys(), default=0), 
                                     max(diagrams2.keys(), default=0))
                        
                        for dim in range(max_dim + 1):
                            dgm1 = diagrams1.get(dim, np.empty((0, 2)))
                            dgm2 = diagrams2.get(dim, np.empty((0, 2)))
                            
                            dist = self.distance_calculator.compute_distance(
                                dgm1, dgm2, metric=metric, **kwargs
                            )
                            total_distance += dist
                        
                        distances[metric] = total_distance
                    
                    # Store in matrices (symmetric)
                    for metric in metrics:
                        dist = distances[metric]
                        self.distance_matrices[metric][i, j] = dist
                        self.distance_matrices[metric][j, i] = dist
                    
                    pbar.update(1)
        
        print(f"Computed {len(metrics)} distance metrics for {n_models} models")
    
    def save_distance_matrices(self) -> None:
        """Save computed distance matrices."""
        if not self.comparison_config['output']['save_distance_matrices']:
            return
        
        print("\nSaving distance matrices...")
        format_type = self.comparison_config['output']['distance_matrix_format']
        
        for metric, matrix in self.distance_matrices.items():
            if format_type == 'npy':
                np.save(self.matrices_dir / f"{metric}_distance_matrix.npy", matrix)
            elif format_type == 'npz':
                np.savez_compressed(
                    self.matrices_dir / f"{metric}_distance_matrix.npz",
                    distance_matrix=matrix,
                    model_names=list(self.persistence_diagrams.keys())
                )
            elif format_type == 'csv':
                # Save as CSV with model names
                import pandas as pd
                model_names = list(self.persistence_diagrams.keys())
                df = pd.DataFrame(matrix, index=model_names, columns=model_names)
                df.to_csv(self.matrices_dir / f"{metric}_distance_matrix.csv")
            elif format_type == 'pickle':
                with open(self.matrices_dir / f"{metric}_distance_matrix.pkl", 'wb') as f:
                    pickle.dump({
                        'matrix': matrix,
                        'model_names': list(self.persistence_diagrams.keys())
                    }, f)
    
    def create_visualizations(self) -> None:
        """Create heatmap visualizations and clustering analysis."""
        if not self.comparison_config['visualization']['create_heatmap']:
            return
        
        print("\nCreating visualizations...")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        viz_config = self.comparison_config['visualization']
        
        for metric, matrix in self.distance_matrices.items():
            # Create figure
            fig, ax = plt.subplots(figsize=viz_config['figsize'])
            
            # Create heatmap
            model_names = [name.split('_')[-1] for name in self.persistence_diagrams.keys()]  # Shorten names
            
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
            
            # Set labels
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_yticklabels(model_names, rotation=0)
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
            
            model_names = [name.split('_')[-1] for name in self.persistence_diagrams.keys()]
            
            dendrogram(
                linkage_matrix,
                labels=model_names,
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
        
        summary = {
            'overview': {
                'total_models': len(self.model_paths),
                'models_processed': len(self.persistence_diagrams),
                'metrics_computed': list(self.distance_matrices.keys())
            },
            'distance_statistics': {}
        }
        
        # Compute statistics for each metric
        for metric, matrix in self.distance_matrices.items():
            # Get upper triangular values (excluding diagonal)
            upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
            
            summary['distance_statistics'][metric] = {
                'mean': float(np.mean(upper_tri)),
                'std': float(np.std(upper_tri)),
                'min': float(np.min(upper_tri)),
                'max': float(np.max(upper_tri)),
                'median': float(np.median(upper_tri))
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