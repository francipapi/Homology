"""
Betti Curves Visualization Script

This script creates various plots showing how Betti numbers change across neural network layers.
It loads computed Betti numbers from the homology computation results and generates:
1. Individual Betti curves for each dimension (B0, B1, B2)
2. Combined Betti curves showing all dimensions
3. Network comparison plots
4. Statistical summary plots

Author: Generated for Homology project
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional
import warnings
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.signal import savgol_filter
import pandas as pd

# Set matplotlib backend and style
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Custom color palette for better visualization
CUSTOM_COLORS = {
    'B0': '#2E86AB',  # Blue for components
    'B1': '#A23B72',  # Magenta for loops  
    'B2': '#F18F01',  # Orange for voids
    'mean': '#2D3748',  # Dark gray for mean lines
    'std': '#A0AEC0'   # Light gray for std bands
}

def safe_tight_layout():
    """Safely apply tight_layout with fallback to manual spacing."""
    try:
        plt.tight_layout()
    except (ValueError, RuntimeError) as e:
        # Fallback to manual spacing if tight_layout fails
        plt.subplots_adjust(hspace=0.3, wspace=0.3)


class BettiCurvesVisualizer:
    """Class for creating Betti curve visualizations."""
    
    def __init__(self, input_dir: str = "results/homology", output_dir: str = "results/plots"):
        """
        Initialize the visualizer.
        
        Parameters:
        - input_dir: Directory containing Betti number results
        - output_dir: Directory to save plots
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.betti_data = None
        self.config = None
        self.dimension_names = ['B₀ (Components)', 'B₁ (Loops)', 'B₂ (Voids)']
        self.colors = [CUSTOM_COLORS['B0'], CUSTOM_COLORS['B1'], CUSTOM_COLORS['B2']]
        self.dimension_labels = ['B₀', 'B₁', 'B₂']
        
        # Additional visualization settings
        self.figsize_large = (14, 8)
        self.figsize_medium = (12, 6) 
        self.figsize_small = (10, 6)
        self.alpha_lines = 0.8
        self.alpha_fill = 0.3
        self.marker_size = 6
        self.line_width = 2.5
        
        # Thresholds for handling many networks
        self.max_networks_per_plot = 10  # Maximum networks to show individually
        self.max_networks_for_markers = 20  # Above this, hide markers
        self.summary_threshold = 5  # Above this, show summary statistics
        
    def load_data(self, filename: str = "layer_betti_numbers_ripser_parallel.pt") -> bool:
        """
        Load Betti numbers data from file.
        
        Parameters:
        - filename: Name of the file containing Betti numbers
        
        Returns:
        - Success status
        """
        try:
            data_path = self.input_dir / filename
            if not data_path.exists():
                print(f"ERROR: Betti numbers file not found: {data_path}")
                return False
            
            # Load Betti numbers
            self.betti_data = torch.load(data_path, map_location='cpu')
            
            # Convert to numpy if it's a tensor
            if isinstance(self.betti_data, torch.Tensor):
                self.betti_data = self.betti_data.numpy()
            
            print(f"Loaded Betti numbers with shape: {self.betti_data.shape}")
            print(f"Expected format: [num_networks, num_layers, num_dimensions]")
            
            # Load configuration if available
            config_path = self.input_dir / "homology_config_used_ripser_parallel.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                print(f"Loaded configuration from: {config_path}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load data: {e}")
            return False
    
    def plot_individual_betti_curves(self, save_format: str = 'png', dpi: int = 300, smooth: bool = False):
        """
        Create individual plots for each Betti dimension.
        
        Parameters:
        - save_format: File format for saving plots
        - dpi: Resolution for saved plots
        """
        if self.betti_data is None:
            print("ERROR: No data loaded. Call load_data() first.")
            return
        
        num_networks, num_layers, num_dimensions = self.betti_data.shape
        layer_indices = np.arange(1, num_layers + 1)  # Layer numbering starts from 1
        
        # Create individual plots for each Betti dimension
        for dim in range(num_dimensions):
            fig, ax = plt.subplots(figsize=self.figsize_medium)
            
            # Handle plotting based on number of networks
            if num_networks <= self.max_networks_per_plot:
                # Plot curves for each network (original behavior)
                for net_idx in range(num_networks):
                    betti_values = self.betti_data[net_idx, :, dim]
                    
                    # Adjust visual properties for many networks
                    if num_networks > self.max_networks_for_markers:
                        marker = None
                        markersize = 0
                    else:
                        marker = 'o'
                        markersize = max(2, self.marker_size - (num_networks // 5))
                    
                    alpha = max(0.3, self.alpha_lines - (num_networks / 50))
                    linewidth = max(1, self.line_width - (num_networks / 20))
                    
                    # Apply smoothing if requested
                    if smooth and num_layers > 4:
                        try:
                            window_length = min(5, num_layers if num_layers % 2 == 1 else num_layers - 1)
                            betti_values_smooth = savgol_filter(betti_values, window_length, 2)
                            ax.plot(layer_indices, betti_values_smooth, 
                                   linewidth=linewidth, alpha=alpha,
                                   label=f'Network {net_idx + 1}' if num_networks <= 5 else None,
                                   color=plt.cm.tab20(net_idx % 20))
                            if marker:
                                ax.plot(layer_indices, betti_values, 
                                       marker=marker, linewidth=0.5, markersize=markersize-1,
                                       alpha=alpha*0.5, color=plt.cm.tab20(net_idx % 20), linestyle='')
                        except:
                            ax.plot(layer_indices, betti_values, 
                                   marker=marker, linewidth=linewidth, markersize=markersize,
                                   label=f'Network {net_idx + 1}' if num_networks <= 5 else None, 
                                   alpha=alpha, color=plt.cm.tab20(net_idx % 20))
                    else:
                        ax.plot(layer_indices, betti_values, 
                               marker=marker, linewidth=linewidth, markersize=markersize,
                               label=f'Network {net_idx + 1}' if num_networks <= 5 else None, 
                               alpha=alpha, color=plt.cm.tab20(net_idx % 20))
            else:
                # For many networks, show only summary statistics
                betti_percentiles = np.percentile(self.betti_data[:, :, dim], [10, 25, 50, 75, 90], axis=0)
                
                # Plot median
                ax.plot(layer_indices, betti_percentiles[2], 
                       color=self.colors[dim], linewidth=3, label='Median', zorder=10)
                
                # Plot quartiles
                ax.fill_between(layer_indices, betti_percentiles[1], betti_percentiles[3],
                               alpha=0.3, color=self.colors[dim], label='IQR (25-75%)')
                
                # Plot 10-90 percentile range
                ax.fill_between(layer_indices, betti_percentiles[0], betti_percentiles[4],
                               alpha=0.15, color=self.colors[dim], label='10-90%')
                
                # Add individual network samples (subset)
                sample_size = min(10, num_networks)
                sample_indices = np.linspace(0, num_networks-1, sample_size, dtype=int)
                for idx in sample_indices:
                    betti_values = self.betti_data[idx, :, dim]
                    ax.plot(layer_indices, betti_values, 
                           linewidth=0.5, alpha=0.3, color='gray', zorder=1)
            
            # Add statistical information
            mean_values = np.mean(self.betti_data[:, :, dim], axis=0)
            std_values = np.std(self.betti_data[:, :, dim], axis=0)
            
            # Add confidence band if multiple networks
            if num_networks > 1:
                ax.fill_between(layer_indices, 
                               np.maximum(0, mean_values - std_values), 
                               mean_values + std_values,
                               alpha=self.alpha_fill, color=CUSTOM_COLORS['std'], 
                               label='±1σ', zorder=0)
                ax.plot(layer_indices, mean_values, 
                       color=CUSTOM_COLORS['mean'], linewidth=self.line_width + 0.5, 
                       label='Mean', alpha=0.9, linestyle='-.')
            
            # Customize plot with improved styling
            ax.set_xlabel('Layer Number', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{self.dimension_names[dim]}', fontsize=14, fontweight='bold', color=self.colors[dim])
            ax.set_title(f'Evolution of {self.dimension_names[dim]} Across Network Layers', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Improved grid and styling
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#FAFBFC')
            
            # Enhanced legend with adaptive positioning
            if num_networks <= self.summary_threshold:
                legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
            else:
                # For many networks, use a simpler legend
                legend = ax.legend(fontsize=10, frameon=True, loc='best')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            
            # Set integer ticks for layers
            ax.set_xticks(layer_indices)
            ax.set_xlim(0.5, num_layers + 0.5)
            
            # Improved y-axis formatting
            ax.set_ylim(bottom=0)
            y_max = int(np.max(self.betti_data[:, :, dim])) + 1
            if y_max <= 10:
                ax.set_yticks(range(0, y_max + 1))
            else:
                ax.set_yticks(np.linspace(0, y_max, min(11, y_max + 1)).astype(int))
            
            # Add summary statistics as text box
            if num_networks > 1:
                if num_networks > self.max_networks_per_plot:
                    stats_text = f'Networks: {num_networks}\nMean: {np.mean(mean_values):.2f}\nStd: {np.mean(std_values):.2f}\nRange: [{np.min(self.betti_data[:,:,dim])}, {np.max(self.betti_data[:,:,dim])}]'
                else:
                    stats_text = f'Mean: {np.mean(mean_values):.2f}\nStd: {np.mean(std_values):.2f}\nRange: [{np.min(self.betti_data[:,:,dim])}, {np.max(self.betti_data[:,:,dim])}]'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10)
            
            safe_tight_layout()
            
            # Save plot
            filename = f'betti_curve_B{dim}_{self.dimension_labels[dim]}'
            save_path = self.output_dir / f"{filename}.{save_format}"
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Enhanced plot saved: {save_path}")
            plt.close()
    
    def _plot_combined_summary(self, ax, layer_indices, num_layers, num_dimensions, normalize=False):
        """Helper method to plot summary statistics for many networks."""
        for dim in range(num_dimensions):
            all_values = self.betti_data[:, :, dim]
            
            if normalize and np.max(all_values) > 0:
                all_values = all_values / np.max(all_values)
            
            # Calculate percentiles
            percentiles = np.percentile(all_values, [10, 25, 50, 75, 90], axis=0)
            
            # Plot median
            ax.plot(layer_indices, percentiles[2], 
                   color=self.colors[dim], linewidth=3, 
                   label=f'{self.dimension_names[dim]} (Median)')
            
            # Plot IQR
            ax.fill_between(layer_indices, percentiles[1], percentiles[3],
                           alpha=0.3, color=self.colors[dim])
            
            # Plot 10-90 range
            ax.fill_between(layer_indices, percentiles[0], percentiles[4],
                           alpha=0.15, color=self.colors[dim])
        
        ax.set_ylabel('Betti Numbers' + (' (Normalized)' if normalize else ''), 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#FAFBFC')
        ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        ax.set_xlim(0.5, num_layers + 0.5)
        ax.set_xticks(layer_indices)
        
        if not normalize:
            ax.set_ylim(bottom=0)
    
    def plot_combined_betti_curves(self, save_format: str = 'png', dpi: int = 300, normalize: bool = False):
        """
        Create combined plots showing all Betti dimensions.
        
        Parameters:
        - save_format: File format for saving plots
        - dpi: Resolution for saved plots
        """
        if self.betti_data is None:
            print("ERROR: No data loaded. Call load_data() first.")
            return
        
        num_networks, num_layers, num_dimensions = self.betti_data.shape
        layer_indices = np.arange(1, num_layers + 1)
        
        # Create enhanced combined visualization
        if num_networks == 1:
            # Single network: create one comprehensive plot
            fig, ax = plt.subplots(figsize=self.figsize_large)
            
            net_idx = 0
            for dim in range(num_dimensions):
                betti_values = self.betti_data[net_idx, :, dim]
                
                # Normalize if requested
                if normalize and np.max(betti_values) > 0:
                    betti_values = betti_values / np.max(betti_values)
                
                ax.plot(layer_indices, betti_values, 
                       marker='o', linewidth=self.line_width, markersize=self.marker_size,
                       label=self.dimension_names[dim], 
                       color=self.colors[dim], alpha=self.alpha_lines)
                
                # Add trend line for better visualization
                if num_layers > 3:
                    z = np.polyfit(layer_indices, betti_values, 1)
                    p = np.poly1d(z)
                    ax.plot(layer_indices, p(layer_indices), 
                           color=self.colors[dim], alpha=0.4, linestyle='--', linewidth=1)
            
            # Enhanced styling
            ax.set_ylabel('Betti Numbers' + (' (Normalized)' if normalize else ''), 
                         fontsize=14, fontweight='bold')
            ax.set_title('Topological Evolution Across Network Layers', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#FAFBFC')
            
            # Enhanced legend with better positioning
            legend = ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, 
                             loc='upper right')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)
            
            ax.set_xlim(0.5, num_layers + 0.5)
            ax.set_xticks(layer_indices)
            ax.set_xlabel('Layer Number', fontsize=14, fontweight='bold')
            
            if not normalize:
                ax.set_ylim(bottom=0)
                y_max = int(np.max(self.betti_data[0, :, :])) + 1
                if y_max <= 10:
                    ax.set_yticks(range(0, y_max + 1))
            else:
                ax.set_ylim(0, 1.1)
                ax.set_yticks(np.linspace(0, 1, 6))
            
        else:
            # Multiple networks: decide visualization strategy based on count
            if num_networks <= 8:
                # For reasonable number of networks, show individual subplots
                max_height = min(5 * num_networks, 20)  # Limit total height to 20 inches
                subplot_height = max_height / num_networks
                fig, axes = plt.subplots(num_networks, 1, figsize=(14, max_height), 
                                        sharex=True, sharey=False)
            else:
                # For many networks, use a grid layout or summary view
                if num_networks <= 20:
                    # Grid layout for up to 20 networks
                    n_cols = 4
                    n_rows = (num_networks + n_cols - 1) // n_cols
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), 
                                           sharex=True, sharey=False)
                    axes = axes.flatten()[:num_networks]  # Only use needed subplots
                else:
                    # For very many networks, create summary plot only
                    fig, ax = plt.subplots(figsize=self.figsize_large)
                    self._plot_combined_summary(ax, layer_indices, num_layers, num_dimensions, normalize)
                    ax.set_xlabel('Layer Number', fontsize=14, fontweight='bold')
                    ax.set_title(f'Topological Evolution Summary ({num_networks} Networks)', 
                               fontsize=16, fontweight='bold', pad=20)
                    safe_tight_layout()
                    suffix = '_normalized' if normalize else ''
                    save_path = self.output_dir / f"betti_curves_combined{suffix}.{save_format}"
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                    print(f"Enhanced combined plot saved: {save_path}")
                    plt.close()
                    return
            
            if num_networks == 1:
                axes = [axes]
            
            for net_idx in range(num_networks):
                ax = axes[net_idx]
                
                # Plot each Betti dimension with enhanced styling
                for dim in range(num_dimensions):
                    betti_values = self.betti_data[net_idx, :, dim]
                    
                    if normalize and np.max(betti_values) > 0:
                        betti_values = betti_values / np.max(betti_values)
                    
                    ax.plot(layer_indices, betti_values, 
                           marker='o', linewidth=self.line_width, markersize=self.marker_size,
                           label=self.dimension_names[dim], 
                           color=self.colors[dim], alpha=self.alpha_lines)
                
                # Enhanced subplot styling
                ax.set_ylabel('Betti Numbers' + (' (Norm.)' if normalize else ''), 
                             fontsize=12, fontweight='bold')
                ax.set_title(f'Network {net_idx + 1}: Topological Features', 
                            fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                ax.set_facecolor('#FAFBFC')
                
                legend = ax.legend(fontsize=10, frameon=True, fancybox=True)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
                
                ax.set_xlim(0.5, num_layers + 0.5)
                ax.set_xticks(layer_indices)
                
                if not normalize:
                    ax.set_ylim(bottom=0)
                    y_max = int(np.max(self.betti_data[net_idx, :, :])) + 1
                    if y_max <= 10:
                        ax.set_yticks(range(0, y_max + 1))
                else:
                    ax.set_ylim(0, 1.1)
            
            # Set x-label only for bottom subplot
            axes[-1].set_xlabel('Layer Number', fontsize=14, fontweight='bold')
        
        safe_tight_layout()
        
        # Save plot with enhanced naming
        suffix = '_normalized' if normalize else ''
        save_path = self.output_dir / f"betti_curves_combined{suffix}.{save_format}"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Enhanced combined plot saved: {save_path}")
        plt.close()
    
    def plot_network_comparison(self, save_format: str = 'png', dpi: int = 300):
        """
        Create comparison plots between networks.
        
        Parameters:
        - save_format: File format for saving plots
        - dpi: Resolution for saved plots
        """
        if self.betti_data is None:
            print("ERROR: No data loaded. Call load_data() first.")
            return
        
        num_networks, num_layers, num_dimensions = self.betti_data.shape
        
        if num_networks < 2:
            print("INFO: Skipping network comparison (only one network available)")
            return
        
        layer_indices = np.arange(1, num_layers + 1)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, num_dimensions, figsize=(5 * num_dimensions, 6))
        
        # Handle single dimension case
        if num_dimensions == 1:
            axes = [axes]
        
        for dim in range(num_dimensions):
            ax = axes[dim]
            
            if num_networks <= self.max_networks_per_plot:
                # Plot each network individually
                for net_idx in range(num_networks):
                    betti_values = self.betti_data[net_idx, :, dim]
                    
                    # Adjust visual properties
                    if num_networks > self.max_networks_for_markers:
                        marker = None
                    else:
                        marker = 'o'
                    
                    alpha = max(0.3, 0.8 - (num_networks / 30))
                    linewidth = max(1, 2 - (num_networks / 20))
                    markersize = max(2, 4 - (num_networks // 10))
                    
                    ax.plot(layer_indices, betti_values, 
                           marker=marker, linewidth=linewidth, markersize=markersize,
                           label=f'Network {net_idx + 1}' if num_networks <= 5 else None, 
                           alpha=alpha, color=plt.cm.tab20(net_idx % 20))
            else:
                # Show statistical summary for many networks
                betti_data_dim = self.betti_data[:, :, dim]
                
                # Plot percentiles
                percentiles = np.percentile(betti_data_dim, [10, 25, 50, 75, 90], axis=0)
                
                ax.plot(layer_indices, percentiles[2], 
                       color=self.colors[dim], linewidth=3, label='Median', zorder=10)
                
                ax.fill_between(layer_indices, percentiles[1], percentiles[3],
                               alpha=0.3, color=self.colors[dim], label='IQR')
                
                ax.fill_between(layer_indices, percentiles[0], percentiles[4],
                               alpha=0.15, color=self.colors[dim], label='10-90%')
                
                # Add text about number of networks
                ax.text(0.02, 0.98, f'{num_networks} Networks', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Customize subplot
            ax.set_xlabel('Layer Number', fontsize=11, fontweight='bold')
            ax.set_ylabel('Betti Numbers', fontsize=11, fontweight='bold')
            ax.set_title(f'{self.dimension_names[dim]}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if num_networks <= self.summary_threshold or num_networks > self.max_networks_per_plot:
                ax.legend(fontsize=9)
            
            ax.set_xlim(0.5, num_layers + 0.5)
            ax.set_ylim(bottom=0)
            
            # Set integer ticks
            ax.set_xticks(layer_indices)
            y_max = int(np.max(self.betti_data[:, :, dim])) + 1
            if y_max <= 10:
                ax.set_yticks(range(0, y_max + 1))
        
        safe_tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"betti_network_comparison.{save_format}"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_topological_phase_space(self, save_format: str = 'png', dpi: int = 300):
        """
        Create phase space plots showing relationships between different Betti numbers.
        """
        if self.betti_data is None or self.betti_data.shape[2] < 2:
            print("INFO: Skipping phase space plot (need at least 2 Betti dimensions)")
            return
        
        num_networks, num_layers, num_dimensions = self.betti_data.shape
        
        # Create phase space plots for each pair of dimensions
        num_plots = min(3, num_dimensions-1)
        fig_width = min(5 * num_plots, 15)
        fig, axes = plt.subplots(1, num_plots, figsize=(fig_width, 5))
        if num_dimensions == 2:
            axes = [axes]
        
        plot_idx = 0
        for dim1 in range(min(num_dimensions-1, 3)):
            dim2 = dim1 + 1
            ax = axes[plot_idx] if len(axes) > 1 else axes
            
            # Handle plotting based on number of networks
            if num_networks <= 10:
                # Plot trajectory for each network
                for net_idx in range(num_networks):
                    x_data = self.betti_data[net_idx, :, dim1]
                    y_data = self.betti_data[net_idx, :, dim2]
                    
                    # Adjust visual properties for many networks
                    alpha = max(0.3, 0.7 - (num_networks / 20))
                    linewidth = max(1, 2 - (num_networks / 10))
                    markersize = max(3, 6 - (num_networks // 5))
                    
                    # Plot trajectory
                    ax.plot(x_data, y_data, 'o-', alpha=alpha, linewidth=linewidth, 
                           markersize=markersize, label=f'Network {net_idx + 1}' if num_networks <= 5 else None,
                           color=plt.cm.tab10(net_idx % 10))
                    
                    if num_networks <= 5:
                        # Add direction arrows for fewer networks
                        for i in range(len(x_data) - 1):
                            dx = x_data[i+1] - x_data[i]
                            dy = y_data[i+1] - y_data[i]
                            if dx != 0 or dy != 0:
                                ax.annotate('', xy=(x_data[i+1], y_data[i+1]), xytext=(x_data[i], y_data[i]),
                                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.5))
                    
                    # Label start and end points
                    ax.scatter(x_data[0], y_data[0], s=50, marker='s', 
                              color='green', alpha=0.6, zorder=5)
                    ax.scatter(x_data[-1], y_data[-1], s=50, marker='^', 
                              color='red', alpha=0.6, zorder=5)
            else:
                # For many networks, show density plot
                all_x = self.betti_data[:, :, dim1].flatten()
                all_y = self.betti_data[:, :, dim2].flatten()
                
                # Create 2D histogram
                from matplotlib.colors import LogNorm
                h = ax.hist2d(all_x, all_y, bins=30, cmap='Blues', norm=LogNorm())
                plt.colorbar(h[3], ax=ax, label='Count (log scale)')
                
                # Add mean trajectory
                mean_x = np.mean(self.betti_data[:, :, dim1], axis=0)
                mean_y = np.mean(self.betti_data[:, :, dim2], axis=0)
                ax.plot(mean_x, mean_y, 'r-', linewidth=3, label='Mean trajectory', zorder=10)
                
                # Mark start and end of mean trajectory
                ax.scatter(mean_x[0], mean_y[0], s=100, marker='s', 
                          color='green', edgecolor='black', linewidth=2, zorder=15)
                ax.scatter(mean_x[-1], mean_y[-1], s=100, marker='^', 
                          color='red', edgecolor='black', linewidth=2, zorder=15)
            
            ax.set_xlabel(f'{self.dimension_names[dim1]}', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{self.dimension_names[dim2]}', fontsize=12, fontweight='bold')
            ax.set_title(f'Topological Phase Space: {self.dimension_labels[dim1]} vs {self.dimension_labels[dim2]}', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.4)
            if num_networks <= 5:
                ax.legend(fontsize=10)
            
            plot_idx += 1
        
        # Add legend for start/end markers only if not too many networks
        if num_networks <= 10:
            legend_elements = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                                         markersize=8, label='Start (Layer 1)'),
                              plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                                         markersize=8, label='End (Final Layer)')]
            if isinstance(axes, list) or hasattr(axes, '__len__'):
                axes[0].legend(handles=legend_elements, loc='upper left', fontsize=9)
            else:
                axes.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        safe_tight_layout()
        save_path = self.output_dir / f"betti_phase_space.{save_format}"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Phase space plot saved: {save_path}")
        plt.close()
    
    def plot_betti_distribution(self, save_format: str = 'png', dpi: int = 300):
        """
        Create distribution plots showing histograms and box plots of Betti numbers.
        """
        if self.betti_data is None:
            print("ERROR: No data loaded. Call load_data() first.")
            return
        
        num_networks, num_layers, num_dimensions = self.betti_data.shape
        
        # Limit figure size for large dimensions
        fig_width = min(5 * num_dimensions, 15)
        fig, axes = plt.subplots(2, num_dimensions, figsize=(fig_width, 10))
        if num_dimensions == 1:
            axes = axes.reshape(-1, 1)
        
        for dim in range(num_dimensions):
            # Flatten data for this dimension
            dim_data = self.betti_data[:, :, dim].flatten()
            
            # Histogram
            ax_hist = axes[0, dim]
            ax_hist.hist(dim_data, bins=max(10, int(np.sqrt(len(dim_data)))), 
                        alpha=0.7, color=self.colors[dim], edgecolor='black', linewidth=0.5)
            ax_hist.set_xlabel(f'{self.dimension_names[dim]}', fontsize=12, fontweight='bold')
            ax_hist.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax_hist.set_title(f'Distribution of {self.dimension_names[dim]}', fontsize=13, fontweight='bold')
            ax_hist.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = np.mean(dim_data)
            std_val = np.std(dim_data)
            stats_text = f'μ = {mean_val:.2f}\nσ = {std_val:.2f}'
            ax_hist.text(0.98, 0.98, stats_text, transform=ax_hist.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Box plot by layer
            ax_box = axes[1, dim]
            layer_data = [self.betti_data[:, layer, dim] for layer in range(num_layers)]
            box_plot = ax_box.boxplot(layer_data, patch_artist=True, labels=range(1, num_layers + 1))
            
            # Color the boxes
            for patch in box_plot['boxes']:
                patch.set_facecolor(self.colors[dim])
                patch.set_alpha(0.7)
            
            ax_box.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
            ax_box.set_ylabel(f'{self.dimension_names[dim]}', fontsize=12, fontweight='bold')
            ax_box.set_title(f'{self.dimension_names[dim]} Distribution by Layer', fontsize=13, fontweight='bold')
            ax_box.grid(True, alpha=0.3)
        
        safe_tight_layout()
        save_path = self.output_dir / f"betti_distributions.{save_format}"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Distribution plots saved: {save_path}")
        plt.close()
    
    def plot_betti_correlation_heatmap(self, save_format: str = 'png', dpi: int = 300):
        """
        Create correlation heatmap between layers and dimensions.
        """
        if self.betti_data is None:
            print("ERROR: No data loaded. Call load_data() first.")
            return
        
        num_networks, num_layers, num_dimensions = self.betti_data.shape
        
        # Create correlation matrix between all layer-dimension combinations
        reshaped_data = self.betti_data.reshape(num_networks, -1)  # Shape: (networks, layers*dimensions)
        correlation_matrix = np.corrcoef(reshaped_data.T)  # Transpose to get correlations between features
        
        # Create labels for the heatmap
        labels = []
        for layer in range(num_layers):
            for dim in range(num_dimensions):
                labels.append(f'L{layer+1}_{self.dimension_labels[dim]}')
        
        # Limit figure size for large datasets
        max_size = min(12, 2 + 0.3 * len(labels))
        fig, ax = plt.subplots(figsize=(max_size, max_size))
        
        # Create heatmap with custom colormap
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # Add correlation values as text
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                              ha='center', va='center', 
                              color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black',
                              fontsize=8)
        
        ax.set_title('Correlation Matrix: Betti Numbers Across Layers', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')
        
        safe_tight_layout()
        save_path = self.output_dir / f"betti_correlation_heatmap.{save_format}"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Correlation heatmap saved: {save_path}")
        plt.close()
    
    def plot_statistical_summary(self, save_format: str = 'png', dpi: int = 300):
        """
        Create statistical summary plots.
        
        Parameters:
        - save_format: File format for saving plots
        - dpi: Resolution for saved plots
        """
        if self.betti_data is None:
            print("ERROR: No data loaded. Call load_data() first.")
            return
        
        num_networks, num_layers, num_dimensions = self.betti_data.shape
        layer_indices = np.arange(1, num_layers + 1)
        
        # Create statistical summary
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Mean and std across networks for each dimension
        ax1 = axes[0, 0]
        for dim in range(num_dimensions):
            mean_values = np.mean(self.betti_data[:, :, dim], axis=0)
            std_values = np.std(self.betti_data[:, :, dim], axis=0)
            
            ax1.errorbar(layer_indices, mean_values, yerr=std_values,
                        marker='o', linewidth=2, markersize=4, capsize=3,
                        label=self.dimension_names[dim], color=self.colors[dim])
        
        ax1.set_xlabel('Layer Number', fontweight='bold')
        ax1.set_ylabel('Mean Betti Numbers ± Std', fontweight='bold')
        ax1.set_title('Statistical Summary Across Networks', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0.5, num_layers + 0.5)
        ax1.set_ylim(bottom=0)
        
        # 2. Total topological complexity per layer
        ax2 = axes[0, 1]
        total_betti = np.sum(self.betti_data, axis=2)  # Sum across dimensions
        
        for net_idx in range(num_networks):
            ax2.plot(layer_indices, total_betti[net_idx, :], 
                    marker='o', linewidth=2, markersize=4,
                    label=f'Network {net_idx + 1}', alpha=0.8)
        
        if num_networks > 1:
            mean_total = np.mean(total_betti, axis=0)
            ax2.plot(layer_indices, mean_total, 'k--', linewidth=2, 
                    label='Mean', alpha=0.7)
        
        ax2.set_xlabel('Layer Number', fontweight='bold')
        ax2.set_ylabel('Total Betti Numbers', fontweight='bold')
        ax2.set_title('Topological Complexity per Layer', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(0.5, num_layers + 0.5)
        ax2.set_ylim(bottom=0)
        
        # 3. Heatmap of Betti numbers (averaged across networks)
        ax3 = axes[1, 0]
        if num_networks > 1:
            heatmap_data = np.mean(self.betti_data, axis=0)  # Average across networks
        else:
            heatmap_data = self.betti_data[0, :, :]
        
        im = ax3.imshow(heatmap_data.T, cmap='viridis', aspect='auto', 
                       interpolation='nearest')
        
        # Set ticks and labels
        ax3.set_xticks(range(num_layers))
        ax3.set_xticklabels([f'L{i+1}' for i in range(num_layers)])
        ax3.set_yticks(range(num_dimensions))
        ax3.set_yticklabels([f'B{i}' for i in range(num_dimensions)])
        ax3.set_xlabel('Layer Number', fontweight='bold')
        ax3.set_ylabel('Betti Dimension', fontweight='bold')
        ax3.set_title('Betti Numbers Heatmap', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Betti Numbers', fontweight='bold')
        
        # Add text annotations
        for i in range(num_layers):
            for j in range(num_dimensions):
                text = ax3.text(i, j, f'{heatmap_data[i, j]:.1f}',
                               ha="center", va="center", color="white", fontweight='bold')
        
        # 4. Layer-to-layer changes
        ax4 = axes[1, 1]
        if num_layers > 1:
            changes = np.diff(self.betti_data, axis=1)  # Layer-to-layer differences
            
            for dim in range(num_dimensions):
                change_values = np.mean(changes[:, :, dim], axis=0)  # Average across networks
                layer_transitions = np.arange(1.5, num_layers + 0.5)  # Between layers
                
                ax4.plot(layer_transitions, change_values, 
                        marker='s', linewidth=2, markersize=4,
                        label=self.dimension_names[dim], color=self.colors[dim])
            
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Layer Transition', fontweight='bold')
            ax4.set_ylabel('Change in Betti Numbers', fontweight='bold')
            ax4.set_title('Layer-to-Layer Changes', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_xlim(1, num_layers)
        else:
            ax4.text(0.5, 0.5, 'Need >1 layer\nfor change analysis', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, fontweight='bold')
            ax4.set_title('Layer-to-Layer Changes', fontweight='bold')
        
        safe_tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"betti_statistical_summary.{save_format}"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def create_summary_report(self):
        """Create a text summary report of the Betti numbers analysis."""
        if self.betti_data is None:
            print("ERROR: No data loaded. Call load_data() first.")
            return
        
        num_networks, num_layers, num_dimensions = self.betti_data.shape
        
        report = []
        report.append("BETTI NUMBERS ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Basic info
        report.append(f"Dataset Information:")
        report.append(f"  Number of networks: {num_networks}")
        report.append(f"  Number of layers: {num_layers}")
        report.append(f"  Betti dimensions: {num_dimensions} (B0, B1, B2)")
        report.append("")
        
        # Statistics for each dimension
        for dim in range(num_dimensions):
            dim_data = self.betti_data[:, :, dim]
            report.append(f"{self.dimension_names[dim]}:")
            report.append(f"  Overall mean: {np.mean(dim_data):.2f}")
            report.append(f"  Overall std:  {np.std(dim_data):.2f}")
            report.append(f"  Min value:    {np.min(dim_data)}")
            report.append(f"  Max value:    {np.max(dim_data)}")
            report.append("")
        
        # Layer-wise analysis (summarized for many networks)
        if num_networks <= 10:
            report.append("Layer-wise Analysis:")
            for layer in range(num_layers):
                layer_data = self.betti_data[:, layer, :]
                total_complexity = np.sum(layer_data, axis=1)
                report.append(f"  Layer {layer + 1}:")
                report.append(f"    Mean total complexity: {np.mean(total_complexity):.2f}")
                report.append(f"    Betti numbers: {np.mean(layer_data, axis=0)}")
            report.append("")
        else:
            # For many networks, provide summary statistics
            report.append("Summary Statistics Across All Layers:")
            total_complexity = np.sum(self.betti_data, axis=2)
            report.append(f"  Mean total complexity: {np.mean(total_complexity):.2f}")
            report.append(f"  Std total complexity: {np.std(total_complexity):.2f}")
            report.append(f"  Max total complexity: {np.max(total_complexity)}")
            report.append(f"  Min total complexity: {np.min(total_complexity)}")
            report.append("")
            
            # Show first and last layer statistics
            report.append("First Layer Statistics:")
            first_layer = self.betti_data[:, 0, :]
            report.append(f"  Mean Betti numbers: {np.mean(first_layer, axis=0)}")
            report.append(f"  Std Betti numbers: {np.std(first_layer, axis=0)}")
            report.append("")
            
            report.append("Final Layer Statistics:")
            final_layer = self.betti_data[:, -1, :]
            report.append(f"  Mean Betti numbers: {np.mean(final_layer, axis=0)}")
            report.append(f"  Std Betti numbers: {np.std(final_layer, axis=0)}")
            report.append("")
        
        # Configuration info
        if self.config:
            report.append("Configuration Used:")
            if 'sampling' in self.config:
                fps_points = self.config['sampling'].get('fps_num_points', 'N/A')
                report.append(f"  Sample points per layer: {fps_points}")
            if 'computation' in self.config:
                max_dim = self.config['computation'].get('max_dimension', 'N/A')
                report.append(f"  Max homology dimension: {max_dim}")
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / "betti_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"Saved analysis report: {report_path}")
        
        # Also print to console
        print("\n" + report_text)
    
    def create_all_plots(self, save_format: str = 'png', dpi: int = 300, 
                        include_advanced: bool = True, smooth_curves: bool = False):
        """
        Create all available plots.
        
        Parameters:
        - save_format: File format for saving plots
        - dpi: Resolution for saved plots
        - include_advanced: Whether to create advanced visualizations
        - smooth_curves: Whether to apply smoothing to individual curves
        """
        print("Creating Enhanced Betti Curves Visualizations...")
        print(f"Output directory: {self.output_dir}")
        print(f"Format: {save_format.upper()}, DPI: {dpi}")
        print("=" * 60)
        
        # Core plots
        print("📊 Creating core visualizations...")
        self.plot_individual_betti_curves(save_format, dpi, smooth=smooth_curves)
        self.plot_combined_betti_curves(save_format, dpi, normalize=False)
        self.plot_combined_betti_curves(save_format, dpi, normalize=True)
        self.plot_network_comparison(save_format, dpi)
        self.plot_statistical_summary(save_format, dpi)
        
        # Advanced plots
        if include_advanced:
            print("🔬 Creating advanced visualizations...")
            self.plot_topological_phase_space(save_format, dpi)
            self.plot_betti_distribution(save_format, dpi)
            self.plot_betti_correlation_heatmap(save_format, dpi)
        
        # Generate comprehensive report
        print("📝 Generating analysis report...")
        self.create_summary_report()
        
        print("=" * 60)
        print("✅ Enhanced Betti curves visualization completed successfully!")
        print(f"📁 All plots saved to: {self.output_dir}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Create Enhanced Betti Curves Visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", type=str, default="results/homology",
                       help="Directory containing Betti numbers data")
    parser.add_argument("--output_dir", type=str, default="results/plots",
                       help="Directory to save plots")
    parser.add_argument("--filename", type=str, default="layer_betti_numbers_ripser_parallel.pt",
                       help="Filename of Betti numbers data")
    parser.add_argument("--format", type=str, default="png", choices=['png', 'pdf', 'svg'],
                       help="Output format for plots")
    parser.add_argument("--dpi", type=int, default=300,
                       help="Resolution for saved plots")
    parser.add_argument("--no-advanced", action="store_true",
                       help="Skip advanced visualizations (phase space, distributions, correlations)")
    parser.add_argument("--smooth", action="store_true",
                       help="Apply smoothing to individual Betti curves")
    parser.add_argument("--version", action="version", version="Enhanced Betti Visualizer v2.0")
    
    args = parser.parse_args()
    
    print("🔬 Enhanced Betti Curves Visualization Tool")
    print("=" * 50)
    
    # Create visualizer
    visualizer = BettiCurvesVisualizer(args.input_dir, args.output_dir)
    
    # Load data
    print(f"📂 Loading data from: {args.input_dir}/{args.filename}")
    if not visualizer.load_data(args.filename):
        print("❌ Failed to load data. Exiting.")
        return
    
    # Create all plots
    visualizer.create_all_plots(
        save_format=args.format, 
        dpi=args.dpi,
        include_advanced=not args.no_advanced,
        smooth_curves=args.smooth
    )


if __name__ == "__main__":
    main()