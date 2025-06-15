"""
Parameter Grid Search for Topological Data Analysis

This script performs systematic parameter optimization for k-neighbors and max_edge_length
in persistent homology computation. It uses synthetic datasets from dataset.py and 
distance computation functions to explore the parameter space comprehensively.

Usage:
    python src/utils/parameter_grid_search.py [--config path/to/search_config.yaml]
"""

import numpy as np
import torch
import yaml
import os
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import scipy as sp
import graph_tool as gt
from graph_tool.topology import shortest_distance
import gudhi as gd
import argparse
from datetime import datetime

# Import project modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.dataset import generate
from src.utils.distance_computation import farthest_point_sampling_pytorch


class ParameterGridSearch:
    """
    Main class for parameter grid search optimization.
    
    Systematically explores k-neighbors and max_edge_length parameter space
    to find optimal values for topological analysis.
    """
    
    def __init__(self, config_path: str = "configs/search_config.yaml"):
        """Initialize grid search with configuration."""
        self.config = self.load_config(config_path)
        self.results = {}
        self.datasets = []
        self.output_dir = Path(self.config['output']['output_dir'])
        self.setup_output_directory()
        self.setup_logging()
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_output_directory(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config['output']['create_subdirs']:
            (self.output_dir / 'plots').mkdir(exist_ok=True)
            (self.output_dir / 'data').mkdir(exist_ok=True)
            (self.output_dir / 'logs').mkdir(exist_ok=True)
            (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration."""
        import logging
        
        # Create logger
        self.logger = logging.getLogger('parameter_search')
        self.logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        if self.config['logging']['log_to_console']:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config['logging']['log_to_file']:
            log_file = self.output_dir / 'logs' / self.config['logging']['log_file']
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def generate_datasets(self) -> List[np.ndarray]:
        """
        Generate multiple dataset instances for robustness testing.
        
        Returns:
            List of numpy arrays containing generated and sampled datasets
        """
        self.logger.info("Generating datasets for grid search...")
        
        datasets = []
        num_instances = self.config['data']['num_instances']
        
        for i in range(num_instances):
            self.logger.info(f"Generating dataset instance {i+1}/{num_instances}")
            
            # Set seed for reproducibility
            if self.config['data']['random_seed'] is not None:
                np.random.seed(self.config['data']['random_seed'] + i)
            
            # Generate dataset using dataset.generate()
            dataset_config = self.config['dataset_params']
            torus_params = dataset_config.get('torus_params', {})
            
            X, y = generate(
                n=dataset_config['n_samples'],
                big_radius=torus_params.get('major_radius', 1.0),
                small_radius=torus_params.get('minor_radius', 0.3)
            )
            
            # Apply FPS sampling if enabled
            if self.config['sampling']['use_fps']:
                fps_points = self.config['sampling']['fps_num_points']
                self.logger.debug(f"Applying FPS: {len(X)} -> {fps_points} points")
                
                # Use efficient FPS implementation with device acceleration
                fps_start_time = time.time()
                X_sampled = self.manual_fps_sampling(X, fps_points, device='auto')
                fps_time = time.time() - fps_start_time
                self.logger.debug(f"FPS sampling completed in {fps_time:.3f}s: {len(X)} -> {len(X_sampled)} points")
                
                # Ensure we have minimum required points
                min_points = self.config['sampling']['min_points_threshold']
                if len(X_sampled) < min_points:
                    self.logger.warning(f"Dataset {i} has only {len(X_sampled)} points, below threshold {min_points}")
                    continue
                    
                datasets.append(X_sampled)
            else:
                datasets.append(X)
        
        self.logger.info(f"Generated {len(datasets)} valid datasets")
        self.datasets = datasets
        return datasets
    
    def manual_fps_sampling(self, points: np.ndarray, k: int, device: str = 'auto') -> np.ndarray:
        """
        Efficient FPS implementation matching distance_computation.py for better performance.
        
        Parameters:
            points: Input points array
            k: Number of points to sample
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        
        Returns:
            Sampled points array
        """
        import torch
        
        # Auto-detect best device for acceleration
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        # Convert to torch tensor with proper device placement
        if isinstance(points, np.ndarray):
            points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        elif isinstance(points, torch.Tensor):
            points_tensor = points.float().to(device)
        else:
            points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        N, D = points_tensor.shape
        
        if k >= N:
            return points if isinstance(points, np.ndarray) else points.cpu().numpy()
        
        # Initialize arrays on the correct device
        sampled_indices = torch.zeros(k, dtype=torch.long, device=device)
        distances = torch.full((N,), float('inf'), device=device)
        
        # Randomly select first point
        sampled_indices[0] = torch.randint(0, N, (1,), device=device)
        last_sampled = points_tensor[sampled_indices[0], :]
        
        for i in range(1, k):
            # Compute squared Euclidean distances from last sampled point to all points
            diff = points_tensor - last_sampled.unsqueeze(0)
            dist_sq = torch.sum(diff ** 2, dim=1)
            
            # Update minimum distances
            distances = torch.minimum(distances, dist_sq)
            
            # Select point with maximum distance
            sampled_indices[i] = torch.argmax(distances)
            last_sampled = points_tensor[sampled_indices[i], :]
        
        # Return sampled points as numpy array
        sampled_points = points_tensor[sampled_indices, :]
        return sampled_points.cpu().numpy()
    
    def compute_knn_geodesic_distance(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        Compute k-NN geodesic distance matrix.
        
        Parameters:
            X: Input points array
            k: Number of nearest neighbors
        
        Returns:
            Integer distance matrix
        """
        graph = kneighbors_graph(X, k, mode='connectivity', p=2, n_jobs=-1)
        g = gt.Graph(sp.sparse.lil_matrix(graph), directed=False)
        distance_matrix = shortest_distance(g)
        return np.array(distance_matrix.get_2d_array(), dtype=np.int32)
    
    def compute_persistent_homology(self, distance_matrix: np.ndarray, 
                                   max_edge_length: float) -> Dict[str, Any]:
        """
        Compute persistent homology and extract metrics.
        
        Parameters:
            distance_matrix: Precomputed distance matrix
            max_edge_length: Maximum edge length for Rips complex
        
        Returns:
            Dictionary containing Betti numbers and additional metrics
        """
        try:
            max_dimension = self.config['homology']['max_dimension']
            
            # Create Rips complex
            rips_complex = gd.RipsComplex(
                distance_matrix=distance_matrix.astype(np.float64), 
                max_edge_length=max_edge_length
            )
            
            # Create simplex tree
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
            
            # Optimize if enabled
            if self.config['homology']['collapse_edges']:
                simplex_tree.collapse_edges()
                simplex_tree.expansion(max_dimension + 1)
            
            # Compute persistence
            persistence = simplex_tree.persistence()
            betti_numbers = simplex_tree.betti_numbers()
            
            # Ensure we have Betti numbers for all dimensions
            while len(betti_numbers) <= max_dimension:
                betti_numbers.append(0)
            
            # Additional metrics
            num_simplices = simplex_tree.num_simplices()
            
            return {
                'betti_numbers': betti_numbers[:max_dimension + 1],
                'num_simplices': num_simplices,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Homology computation failed: {e}")
            max_dimension = self.config['homology']['max_dimension']
            return {
                'betti_numbers': [1] + [0] * max_dimension,
                'num_simplices': 0,
                'success': False
            }
    
    def analyze_graph_connectivity(self, X: np.ndarray, k: int) -> Dict[str, float]:
        """
        Analyze connectivity properties of k-NN graph.
        
        Parameters:
            X: Input points
            k: Number of nearest neighbors
        
        Returns:
            Dictionary with connectivity metrics
        """
        try:
            # Build k-NN graph
            graph = kneighbors_graph(X, k, mode='connectivity', p=2, n_jobs=-1)
            g = gt.Graph(sp.sparse.lil_matrix(graph), directed=False)
            
            # Compute connected components
            comp, hist = gt.topology.label_components(g)
            largest_component_size = max(hist) if len(hist) > 0 else 0
            num_components = len(hist)
            
            # Connectivity metrics
            total_nodes = g.num_vertices()
            connectivity_ratio = largest_component_size / total_nodes if total_nodes > 0 else 0
            
            return {
                'num_components': num_components,
                'largest_component_size': largest_component_size,
                'connectivity_ratio': connectivity_ratio,
                'total_nodes': total_nodes
            }
            
        except Exception as e:
            self.logger.error(f"Connectivity analysis failed: {e}")
            return {
                'num_components': len(X),
                'largest_component_size': 1,
                'connectivity_ratio': 1.0 / len(X),
                'total_nodes': len(X)
            }
    
    def run_single_parameter_combination(self, k: int, max_edge_length: float, 
                                       dataset_idx: int) -> Dict[str, Any]:
        """
        Run homology computation for a single parameter combination on one dataset.
        
        Parameters:
            k: Number of nearest neighbors
            max_edge_length: Maximum edge length for Rips complex
            dataset_idx: Index of dataset to use
        
        Returns:
            Dictionary with computation results
        """
        start_time = time.time()
        
        # Get dataset
        X = self.datasets[dataset_idx]
        
        try:
            # Compute distance matrix
            distance_matrix = self.compute_knn_geodesic_distance(X, k)
            
            # Analyze connectivity
            connectivity_metrics = self.analyze_graph_connectivity(X, k)
            
            # Compute persistent homology
            homology_results = self.compute_persistent_homology(distance_matrix, max_edge_length)
            
            # Combine results
            results = {
                'k': k,
                'max_edge_length': max_edge_length,
                'dataset_idx': dataset_idx,
                'betti_numbers': homology_results['betti_numbers'],
                'num_simplices': homology_results['num_simplices'],
                'connectivity': connectivity_metrics,
                'computation_time': time.time() - start_time,
                'success': homology_results['success']
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Parameter combination failed (k={k}, edge_length={max_edge_length}): {e}")
            return {
                'k': k,
                'max_edge_length': max_edge_length,
                'dataset_idx': dataset_idx,
                'betti_numbers': [1, 0, 0],
                'num_simplices': 0,
                'connectivity': {'num_components': len(X), 'connectivity_ratio': 0.0},
                'computation_time': time.time() - start_time,
                'success': False
            }
    
    def run_grid_search(self) -> Dict[str, Any]:
        """
        Execute the complete grid search over all parameter combinations.
        
        Returns:
            Dictionary containing all results
        """
        self.logger.info("Starting parameter grid search...")
        
        # Generate datasets
        if not self.datasets:
            self.generate_datasets()
        
        if not self.datasets:
            raise ValueError("No valid datasets generated!")
        
        # Get parameter grids
        k_values = self.config['grid_search']['k_neighbors']['values']
        edge_lengths = self.config['grid_search']['max_edge_length']['values']
        
        total_combinations = len(k_values) * len(edge_lengths) * len(self.datasets)
        self.logger.info(f"Testing {len(k_values)} k-values × {len(edge_lengths)} edge lengths × {len(self.datasets)} datasets = {total_combinations} combinations")
        
        # Initialize results storage
        all_results = []
        combination_count = 0
        
        # Main grid search loop
        for k in k_values:
            for edge_length in edge_lengths:
                self.logger.info(f"Testing k={k}, max_edge_length={edge_length}")
                
                # Test on all datasets for robustness
                combination_results = []
                for dataset_idx in range(len(self.datasets)):
                    result = self.run_single_parameter_combination(k, edge_length, dataset_idx)
                    combination_results.append(result)
                    combination_count += 1
                    
                    # Progress reporting
                    if self.config['performance']['show_progress'] and \
                       combination_count % self.config['performance']['progress_frequency'] == 0:
                        progress = 100 * combination_count / total_combinations
                        self.logger.info(f"Progress: {progress:.1f}% ({combination_count}/{total_combinations})")
                
                # Store results for this parameter combination
                all_results.extend(combination_results)
                
                # Save checkpoint if enabled
                if self.config['performance']['save_checkpoints'] and \
                   len(all_results) % (self.config['performance']['checkpoint_frequency'] * len(self.datasets)) == 0:
                    self.save_checkpoint(all_results)
        
        # Organize results
        self.results = {
            'raw_results': all_results,
            'k_values': k_values,
            'edge_lengths': edge_lengths,
            'num_datasets': len(self.datasets),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info("Grid search completed successfully!")
        return self.results
    
    def save_checkpoint(self, results: List[Dict]) -> None:
        """Save intermediate results as checkpoint."""
        checkpoint_file = self.output_dir / 'checkpoints' / f'checkpoint_{len(results)}.pkl'
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(results, f)
        self.logger.debug(f"Saved checkpoint with {len(results)} results")
    
    def compute_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics across dataset instances.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.results:
            raise ValueError("No results available. Run grid search first.")
        
        self.logger.info("Computing summary statistics...")
        
        raw_results = self.results['raw_results']
        k_values = self.results['k_values']
        edge_lengths = self.results['edge_lengths']
        max_dimension = self.config['homology']['max_dimension']
        
        # Initialize summary arrays
        summary_stats = {}
        
        # Group results by (k, edge_length)
        for k in k_values:
            for edge_length in edge_lengths:
                # Get results for this parameter combination
                param_results = [r for r in raw_results if r['k'] == k and r['max_edge_length'] == edge_length]
                
                if not param_results:
                    continue
                
                # Extract Betti numbers across datasets
                betti_arrays = np.array([r['betti_numbers'] for r in param_results])
                
                # Compute statistics
                key = (k, edge_length)
                summary_stats[key] = {
                    'k': k,
                    'max_edge_length': edge_length,
                    'betti_mean': np.mean(betti_arrays, axis=0),
                    'betti_std': np.std(betti_arrays, axis=0),
                    'betti_min': np.min(betti_arrays, axis=0),
                    'betti_max': np.max(betti_arrays, axis=0),
                    'connectivity_mean': np.mean([r['connectivity']['connectivity_ratio'] for r in param_results]),
                    'connectivity_std': np.std([r['connectivity']['connectivity_ratio'] for r in param_results]),
                    'computation_time_mean': np.mean([r['computation_time'] for r in param_results]),
                    'computation_time_std': np.std([r['computation_time'] for r in param_results]),
                    'success_rate': np.mean([r['success'] for r in param_results]),
                    'num_datasets': len(param_results)
                }
                
                # Derived metrics
                if self.config['analysis']['compute_complexity']:
                    complexity = np.sum(summary_stats[key]['betti_mean'][1:])  # H1 + H2 + ...
                    summary_stats[key]['topological_complexity'] = complexity
                
                if self.config['analysis']['compute_signal_noise']:
                    # Signal-to-noise ratio for each dimension
                    snr = np.where(summary_stats[key]['betti_std'] > 0,
                                  summary_stats[key]['betti_mean'] / summary_stats[key]['betti_std'],
                                  np.inf)
                    summary_stats[key]['signal_noise_ratio'] = snr
        
        self.results['summary_stats'] = summary_stats
        self.logger.info(f"Computed summary statistics for {len(summary_stats)} parameter combinations")
        
        return summary_stats
    
    def create_visualizations(self) -> None:
        """
        Generate all visualization plots.
        """
        if not self.results or 'summary_stats' not in self.results:
            self.compute_summary_statistics()
        
        self.logger.info("Generating visualizations...")
        
        # Setup plot styling
        plt.style.use('default')
        sns.set_palette("viridis")
        
        viz_config = self.config['visualization']
        
        # Create individual plots
        if viz_config['create_heatmaps']:
            self.create_betti_heatmaps()
        
        if viz_config['create_stability_plots']:
            self.create_stability_plots()
        
        if viz_config['create_connectivity_plots']:
            self.create_connectivity_plots()
        
        if viz_config['create_sensitivity_plots']:
            self.create_sensitivity_plots()
        
        if viz_config['create_timing_plots']:
            self.create_timing_plots()
        
        # Create combined report
        if self.config['output']['save_combined_report']:
            self.create_combined_report()
    
    def create_betti_heatmaps(self) -> None:
        """Create heatmaps for Betti numbers."""
        summary_stats = self.results['summary_stats']
        k_values = self.results['k_values']
        edge_lengths = self.results['edge_lengths']
        max_dimension = self.config['homology']['max_dimension']
        
        fig, axes = plt.subplots(1, max_dimension + 1, figsize=(15, 4))
        if max_dimension == 0:
            axes = [axes]
        
        for dim in range(max_dimension + 1):
            # Create data matrix
            data_matrix = np.zeros((len(edge_lengths), len(k_values)))
            
            for i, edge_length in enumerate(edge_lengths):
                for j, k in enumerate(k_values):
                    key = (k, edge_length)
                    if key in summary_stats:
                        data_matrix[i, j] = summary_stats[key]['betti_mean'][dim]
                    else:
                        data_matrix[i, j] = np.nan
            
            # Create heatmap
            im = axes[dim].imshow(data_matrix, aspect='auto', cmap='plasma', origin='lower')
            axes[dim].set_title(f'H{dim} (Betti {dim})')
            axes[dim].set_xlabel('k-neighbors')
            axes[dim].set_ylabel('Max Edge Length')
            
            # Set ticks
            axes[dim].set_xticks(range(len(k_values)))
            axes[dim].set_xticklabels(k_values)
            axes[dim].set_yticks(range(len(edge_lengths)))
            axes[dim].set_yticklabels(edge_lengths)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[dim])
            
            # Add value annotations if enabled
            if self.config['visualization']['heatmap_annotation']:
                for i in range(len(edge_lengths)):
                    for j in range(len(k_values)):
                        if not np.isnan(data_matrix[i, j]):
                            text = axes[dim].text(j, i, f'{data_matrix[i, j]:.1f}',
                                               ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        if self.config['output']['save_individual_plots']:
            filename = self.output_dir / 'plots' / 'betti_heatmaps.png'
            plt.savefig(filename, dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight')
            self.logger.info(f"Saved Betti heatmaps to {filename}")
        
        plt.close()
    
    def create_stability_plots(self) -> None:
        """Create stability analysis plots."""
        summary_stats = self.results['summary_stats']
        k_values = self.results['k_values']
        edge_lengths = self.results['edge_lengths']
        max_dimension = self.config['homology']['max_dimension']
        
        fig, axes = plt.subplots(1, max_dimension + 1, figsize=(15, 4))
        if max_dimension == 0:
            axes = [axes]
        
        for dim in range(max_dimension + 1):
            # Create stability matrix (standard deviation)
            stability_matrix = np.zeros((len(edge_lengths), len(k_values)))
            
            for i, edge_length in enumerate(edge_lengths):
                for j, k in enumerate(k_values):
                    key = (k, edge_length)
                    if key in summary_stats:
                        stability_matrix[i, j] = summary_stats[key]['betti_std'][dim]
                    else:
                        stability_matrix[i, j] = np.nan
            
            # Create heatmap
            im = axes[dim].imshow(stability_matrix, aspect='auto', cmap='Reds', origin='lower')
            axes[dim].set_title(f'H{dim} Stability (Std Dev)')
            axes[dim].set_xlabel('k-neighbors')
            axes[dim].set_ylabel('Max Edge Length')
            
            # Set ticks
            axes[dim].set_xticks(range(len(k_values)))
            axes[dim].set_xticklabels(k_values)
            axes[dim].set_yticks(range(len(edge_lengths)))
            axes[dim].set_yticklabels(edge_lengths)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[dim])
        
        plt.tight_layout()
        
        # Save plot
        if self.config['output']['save_individual_plots']:
            filename = self.output_dir / 'plots' / 'stability_plots.png'
            plt.savefig(filename, dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight')
            self.logger.info(f"Saved stability plots to {filename}")
        
        plt.close()
    
    def create_connectivity_plots(self) -> None:
        """Create connectivity analysis plots."""
        summary_stats = self.results['summary_stats']
        k_values = self.results['k_values']
        edge_lengths = self.results['edge_lengths']
        
        # Connectivity ratio heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        connectivity_matrix = np.zeros((len(edge_lengths), len(k_values)))
        
        for i, edge_length in enumerate(edge_lengths):
            for j, k in enumerate(k_values):
                key = (k, edge_length)
                if key in summary_stats:
                    connectivity_matrix[i, j] = summary_stats[key]['connectivity_mean']
                else:
                    connectivity_matrix[i, j] = np.nan
        
        im = ax.imshow(connectivity_matrix, aspect='auto', cmap='Blues', origin='lower')
        ax.set_title('Graph Connectivity Ratio')
        ax.set_xlabel('k-neighbors')
        ax.set_ylabel('Max Edge Length')
        
        # Set ticks
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values)
        ax.set_yticks(range(len(edge_lengths)))
        ax.set_yticklabels(edge_lengths)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        # Save plot
        if self.config['output']['save_individual_plots']:
            filename = self.output_dir / 'plots' / 'connectivity_plots.png'
            plt.savefig(filename, dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight')
            self.logger.info(f"Saved connectivity plots to {filename}")
        
        plt.close()
    
    def create_sensitivity_plots(self) -> None:
        """Create parameter sensitivity analysis plots."""
        summary_stats = self.results['summary_stats']
        k_values = self.results['k_values']
        edge_lengths = self.results['edge_lengths']
        max_dimension = self.config['homology']['max_dimension']
        
        fig, axes = plt.subplots(2, max_dimension + 1, figsize=(15, 8))
        if max_dimension == 0:
            axes = axes.reshape(2, 1)
        
        # Plot 1: Betti numbers vs k (for different edge lengths)
        for dim in range(max_dimension + 1):
            ax = axes[0, dim]
            
            for edge_length in edge_lengths[::2]:  # Sample every other edge length
                betti_values = []
                for k in k_values:
                    key = (k, edge_length)
                    if key in summary_stats:
                        betti_values.append(summary_stats[key]['betti_mean'][dim])
                    else:
                        betti_values.append(np.nan)
                
                ax.plot(k_values, betti_values, marker='o', label=f'edge_len={edge_length}')
            
            ax.set_xlabel('k-neighbors')
            ax.set_ylabel(f'H{dim} (Betti {dim})')
            ax.set_title(f'H{dim} vs k-neighbors')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Betti numbers vs edge length (for different k values)
        for dim in range(max_dimension + 1):
            ax = axes[1, dim]
            
            for k in k_values[::2]:  # Sample every other k value
                betti_values = []
                for edge_length in edge_lengths:
                    key = (k, edge_length)
                    if key in summary_stats:
                        betti_values.append(summary_stats[key]['betti_mean'][dim])
                    else:
                        betti_values.append(np.nan)
                
                ax.plot(edge_lengths, betti_values, marker='s', label=f'k={k}')
            
            ax.set_xlabel('Max Edge Length')
            ax.set_ylabel(f'H{dim} (Betti {dim})')
            ax.set_title(f'H{dim} vs Edge Length')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if self.config['output']['save_individual_plots']:
            filename = self.output_dir / 'plots' / 'sensitivity_plots.png'
            plt.savefig(filename, dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight')
            self.logger.info(f"Saved sensitivity plots to {filename}")
        
        plt.close()
    
    def create_timing_plots(self) -> None:
        """Create computational timing analysis plots."""
        summary_stats = self.results['summary_stats']
        k_values = self.results['k_values']
        edge_lengths = self.results['edge_lengths']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        timing_matrix = np.zeros((len(edge_lengths), len(k_values)))
        
        for i, edge_length in enumerate(edge_lengths):
            for j, k in enumerate(k_values):
                key = (k, edge_length)
                if key in summary_stats:
                    timing_matrix[i, j] = summary_stats[key]['computation_time_mean']
                else:
                    timing_matrix[i, j] = np.nan
        
        im = ax.imshow(timing_matrix, aspect='auto', cmap='YlOrRd', origin='lower')
        ax.set_title('Computation Time (seconds)')
        ax.set_xlabel('k-neighbors')
        ax.set_ylabel('Max Edge Length')
        
        # Set ticks
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values)
        ax.set_yticks(range(len(edge_lengths)))
        ax.set_yticklabels(edge_lengths)
        
        # Add value annotations
        for i in range(len(edge_lengths)):
            for j in range(len(k_values)):
                if not np.isnan(timing_matrix[i, j]):
                    text = ax.text(j, i, f'{timing_matrix[i, j]:.1f}s',
                                 ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        # Save plot
        if self.config['output']['save_individual_plots']:
            filename = self.output_dir / 'plots' / 'timing_plots.png'
            plt.savefig(filename, dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight')
            self.logger.info(f"Saved timing plots to {filename}")
        
        plt.close()
    
    def create_combined_report(self) -> None:
        """Create combined PDF report with all visualizations."""
        self.logger.info("Creating combined PDF report...")
        
        pdf_filename = self.output_dir / 'parameter_search_report.pdf'
        
        with PdfPages(pdf_filename) as pdf:
            # Recreate all plots and add to PDF
            self.create_betti_heatmaps()
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            plt.close()
            
            self.create_stability_plots()
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            plt.close()
            
            self.create_connectivity_plots()
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            plt.close()
            
            self.create_sensitivity_plots()
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            plt.close()
            
            self.create_timing_plots()
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Saved combined report to {pdf_filename}")
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate parameter recommendations based on analysis.
        
        Returns:
            List of recommended parameter combinations with justification
        """
        if 'summary_stats' not in self.results:
            self.compute_summary_statistics()
        
        self.logger.info("Generating parameter recommendations...")
        
        summary_stats = self.results['summary_stats']
        rec_config = self.config['recommendation']
        
        # Score each parameter combination
        scored_params = []
        
        for key, stats in summary_stats.items():
            k, edge_length = key
            
            # Apply filtering criteria
            if stats['computation_time_mean'] > rec_config['max_computation_time']:
                continue
            if stats['connectivity_mean'] < rec_config['min_connectivity']:
                continue
            if np.max(stats['betti_std']) > rec_config['max_stability_variance']:
                continue
            
            # Compute composite score
            score = 0.0
            
            # Stability component (lower std is better)
            stability_score = 1.0 / (1.0 + np.mean(stats['betti_std']))
            score += rec_config['stability_weight'] * stability_score
            
            # Connectivity component
            connectivity_score = stats['connectivity_mean']
            score += rec_config['connectivity_weight'] * connectivity_score
            
            # Complexity component (moderate complexity is good)
            if 'topological_complexity' in stats:
                complexity = stats['topological_complexity']
                complexity_score = 1.0 / (1.0 + abs(complexity - 2.0))  # Target complexity ~2
                score += rec_config['complexity_weight'] * complexity_score
            
            # Efficiency component (faster is better)
            efficiency_score = 1.0 / (1.0 + stats['computation_time_mean'])
            score += rec_config['efficiency_weight'] * efficiency_score
            
            scored_params.append({
                'k': k,
                'max_edge_length': edge_length,
                'score': score,
                'stats': stats,
                'justification': self._generate_justification(stats, stability_score, 
                                                           connectivity_score, efficiency_score)
            })
        
        # Sort by score and get top recommendations
        scored_params.sort(key=lambda x: x['score'], reverse=True)
        top_recommendations = scored_params[:rec_config['top_n_recommendations']]
        
        self.results['recommendations'] = top_recommendations
        
        self.logger.info(f"Generated {len(top_recommendations)} parameter recommendations")
        return top_recommendations
    
    def _generate_justification(self, stats: Dict, stability_score: float, 
                              connectivity_score: float, efficiency_score: float) -> str:
        """Generate human-readable justification for parameter recommendation."""
        justifications = []
        
        if stability_score > 0.8:
            justifications.append("high stability across datasets")
        if connectivity_score > 0.9:
            justifications.append("excellent graph connectivity")
        if efficiency_score > 0.5:
            justifications.append("reasonable computational efficiency")
        if stats['success_rate'] == 1.0:
            justifications.append("100% computation success rate")
        
        return "; ".join(justifications) if justifications else "balanced trade-offs"
    
    def save_results(self) -> None:
        """Save all results to files."""
        self.logger.info("Saving results...")
        
        # Save raw results
        if self.config['output']['save_raw_results']:
            if self.config['output']['output_format'] == 'pytorch':
                filename = self.output_dir / 'data' / 'raw_results.pt'
                torch.save(self.results, filename)
            elif self.config['output']['output_format'] == 'numpy':
                filename = self.output_dir / 'data' / 'raw_results.npz'
                np.savez_compressed(filename, **self.results)
            else:  # csv
                # Convert to DataFrame and save
                df = pd.DataFrame(self.results['raw_results'])
                filename = self.output_dir / 'data' / 'raw_results.csv'
                df.to_csv(filename, index=False)
            
            self.logger.info(f"Saved raw results to {filename}")
        
        # Save summary statistics
        if self.config['output']['save_summary_stats'] and 'summary_stats' in self.results:
            filename = self.output_dir / 'data' / 'summary_statistics.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(self.results['summary_stats'], f)
            self.logger.info(f"Saved summary statistics to {filename}")
        
        # Save recommendations
        if self.config['output']['save_recommendations'] and 'recommendations' in self.results:
            filename = self.output_dir / 'data' / 'recommendations.yaml'
            # Convert numpy types to native Python types for YAML serialization
            recommendations_clean = []
            for rec in self.results['recommendations']:
                rec_clean = {
                    'k': int(rec['k']),
                    'max_edge_length': int(rec['max_edge_length']),
                    'score': float(rec['score']),
                    'justification': rec['justification']
                }
                recommendations_clean.append(rec_clean)
            
            with open(filename, 'w') as f:
                yaml.safe_dump(recommendations_clean, f, default_flow_style=False)
            self.logger.info(f"Saved recommendations to {filename}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete parameter grid search analysis pipeline.
        
        Returns:
            Dictionary containing all results and recommendations
        """
        start_time = time.time()
        self.logger.info("Starting complete parameter grid search analysis")
        
        try:
            # Step 1: Grid search
            self.run_grid_search()
            
            # Step 2: Analysis
            self.compute_summary_statistics()
            
            # Step 3: Visualizations
            self.create_visualizations()
            
            # Step 4: Recommendations
            self.generate_recommendations()
            
            # Step 5: Save results
            self.save_results()
            
            total_time = time.time() - start_time
            self.logger.info(f"Complete analysis finished in {total_time:.2f} seconds")
            
            # Print summary
            self.print_summary()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise
    
    def print_summary(self) -> None:
        """Print summary of results and recommendations."""
        print("\n" + "="*60)
        print("PARAMETER GRID SEARCH SUMMARY")
        print("="*60)
        
        if 'recommendations' in self.results:
            print(f"\nTop {len(self.results['recommendations'])} Parameter Recommendations:")
            print("-" * 50)
            
            for i, rec in enumerate(self.results['recommendations'][:3], 1):
                print(f"\n{i}. k={rec['k']}, max_edge_length={rec['max_edge_length']}")
                print(f"   Score: {rec['score']:.3f}")
                print(f"   Justification: {rec['justification']}")
                stats = rec['stats']
                print(f"   Betti numbers (mean): {[f'{x:.1f}' for x in stats['betti_mean']]}")
                print(f"   Stability (std): {[f'{x:.2f}' for x in stats['betti_std']]}")
                print(f"   Connectivity: {stats['connectivity_mean']:.3f}")
                print(f"   Computation time: {stats['computation_time_mean']:.2f}s")
        
        print(f"\nResults saved to: {self.output_dir}")
        print("="*60)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Parameter Grid Search for Topological Data Analysis")
    parser.add_argument('--config', type=str, default='configs/search_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                       help='Override output directory')
    
    args = parser.parse_args()
    
    # Create grid search instance
    search = ParameterGridSearch(config_path=args.config)
    
    # Override output directory if specified
    if args.output:
        search.output_dir = Path(args.output)
        search.setup_output_directory()
    
    # Run complete analysis
    try:
        results = search.run_complete_analysis()
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {search.output_dir}")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        search.logger.info("Analysis interrupted by user")
        
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        search.logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()