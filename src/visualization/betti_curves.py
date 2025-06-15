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

# Set matplotlib backend and style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

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
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
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
    
    def plot_individual_betti_curves(self, save_format: str = 'png', dpi: int = 300):
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
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot curves for each network
            for net_idx in range(num_networks):
                betti_values = self.betti_data[net_idx, :, dim]
                ax.plot(layer_indices, betti_values, 
                       marker='o', linewidth=2, markersize=4,
                       label=f'Network {net_idx + 1}', alpha=0.8)
            
            # Customize plot
            ax.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{self.dimension_names[dim]}', fontsize=12, fontweight='bold')
            ax.set_title(f'Betti Numbers Across Layers: {self.dimension_names[dim]}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Set integer ticks for layers
            ax.set_xticks(layer_indices)
            ax.set_xlim(0.5, num_layers + 0.5)
            
            # Ensure y-axis starts from 0 and uses integers
            ax.set_ylim(bottom=0)
            y_max = int(np.max(self.betti_data[:, :, dim])) + 1
            ax.set_yticks(range(0, y_max + 1))
            
            # Add statistical information
            mean_values = np.mean(self.betti_data[:, :, dim], axis=0)
            std_values = np.std(self.betti_data[:, :, dim], axis=0)
            
            # Add confidence band if multiple networks
            if num_networks > 1:
                ax.fill_between(layer_indices, 
                               mean_values - std_values, 
                               mean_values + std_values,
                               alpha=0.2, color='gray', label='±1 std')
                ax.plot(layer_indices, mean_values, 'k--', linewidth=2, 
                       label='Mean', alpha=0.7)
                ax.legend(fontsize=10)
            
            plt.tight_layout()
            
            # Save plot
            filename = f'betti_curve_B{dim}_{self.dimension_names[dim].split()[0]}'
            save_path = self.output_dir / f"{filename}.{save_format}"
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
    
    def plot_combined_betti_curves(self, save_format: str = 'png', dpi: int = 300):
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
        
        # Create subplots for each network
        fig, axes = plt.subplots(num_networks, 1, figsize=(12, 4 * num_networks), 
                                sharex=True, sharey=False)
        
        # Handle single network case
        if num_networks == 1:
            axes = [axes]
        
        for net_idx in range(num_networks):
            ax = axes[net_idx]
            
            # Plot each Betti dimension
            for dim in range(num_dimensions):
                betti_values = self.betti_data[net_idx, :, dim]
                ax.plot(layer_indices, betti_values, 
                       marker='o', linewidth=2.5, markersize=5,
                       label=self.dimension_names[dim], 
                       color=self.colors[dim], alpha=0.8)
            
            # Customize subplot
            ax.set_ylabel('Betti Numbers', fontsize=11, fontweight='bold')
            ax.set_title(f'Network {net_idx + 1}: All Betti Dimensions', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_xlim(0.5, num_layers + 0.5)
            ax.set_ylim(bottom=0)
            
            # Set integer ticks
            ax.set_xticks(layer_indices)
            y_max = int(np.max(self.betti_data[net_idx, :, :])) + 1
            ax.set_yticks(range(0, y_max + 1))
        
        # Set x-label only for bottom subplot
        axes[-1].set_xlabel('Layer Number', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"betti_curves_combined.{save_format}"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
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
            
            # Plot each network
            for net_idx in range(num_networks):
                betti_values = self.betti_data[net_idx, :, dim]
                ax.plot(layer_indices, betti_values, 
                       marker='o', linewidth=2, markersize=4,
                       label=f'Network {net_idx + 1}', alpha=0.8)
            
            # Customize subplot
            ax.set_xlabel('Layer Number', fontsize=11, fontweight='bold')
            ax.set_ylabel('Betti Numbers', fontsize=11, fontweight='bold')
            ax.set_title(f'{self.dimension_names[dim]}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_xlim(0.5, num_layers + 0.5)
            ax.set_ylim(bottom=0)
            
            # Set integer ticks
            ax.set_xticks(layer_indices)
            y_max = int(np.max(self.betti_data[:, :, dim])) + 1
            ax.set_yticks(range(0, y_max + 1))
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"betti_network_comparison.{save_format}"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
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
        
        plt.tight_layout()
        
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
        
        # Layer-wise analysis
        report.append("Layer-wise Analysis:")
        for layer in range(num_layers):
            layer_data = self.betti_data[:, layer, :]
            total_complexity = np.sum(layer_data, axis=1)
            report.append(f"  Layer {layer + 1}:")
            report.append(f"    Mean total complexity: {np.mean(total_complexity):.2f}")
            report.append(f"    Betti numbers: {np.mean(layer_data, axis=0)}")
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
    
    def create_all_plots(self, save_format: str = 'png', dpi: int = 300):
        """
        Create all available plots.
        
        Parameters:
        - save_format: File format for saving plots
        - dpi: Resolution for saved plots
        """
        print("Creating Betti curves visualizations...")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        # Create all plots
        self.plot_individual_betti_curves(save_format, dpi)
        self.plot_combined_betti_curves(save_format, dpi)
        self.plot_network_comparison(save_format, dpi)
        self.plot_statistical_summary(save_format, dpi)
        self.create_summary_report()
        
        print("-" * 50)
        print("Betti curves visualization completed successfully!")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Create Betti curves visualizations")
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
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = BettiCurvesVisualizer(args.input_dir, args.output_dir)
    
    # Load data
    if not visualizer.load_data(args.filename):
        return
    
    # Create all plots
    visualizer.create_all_plots(args.format, args.dpi)


if __name__ == "__main__":
    main()