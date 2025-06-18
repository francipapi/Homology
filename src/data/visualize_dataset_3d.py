#!/usr/bin/env python3
"""
Interactive 3D Dataset Visualization

This script provides interactive 3D visualization of generated torus datasets,
supporting both hollow and solid tori with configurable parameters.

Features:
- Interactive 3D plotting with rotation and zoom
- Side-by-side comparison of hollow vs solid tori
- Color-coded point clouds by class labels
- Configurable point sizes and transparency
- Save plots to files
- Support for different visualization backends

Usage:
    python src/data/visualize_dataset_3d.py [--config path/to/config.yaml] [--interactive] [--save-plots]
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
import yaml
import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.data.dataset import generate, gen_easy
    print("✅ Successfully imported dataset generation functions")
except ImportError as e:
    print(f"❌ Failed to import dataset functions: {e}")
    sys.exit(1)


class TorusDataset3DVisualizer:
    """
    Interactive 3D visualization tool for torus datasets.
    
    Supports both matplotlib and plotly backends for different use cases.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize visualizer with optional configuration."""
        self.config = self.load_config(config_path) if config_path else self.get_default_config()
        self.data_cache = {}
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded configuration from {config_path}")
            return config
        except Exception as e:
            print(f"⚠️  Failed to load config, using defaults: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default visualization configuration."""
        return {
            'dataset': {
                'n_samples': 2000,
                'big_radius': 3.0,
                'small_radius': 1.0,
                'noise_level': 0.0
            },
            'visualization': {
                'backend': 'plotly',  # 'matplotlib' or 'plotly'
                'figure_size': [12, 8],
                'point_size': 2,
                'alpha': 0.7,
                'color_scheme': 'class_based',  # 'class_based' or 'coordinate_based'
                'show_axes': True,
                'show_grid': True
            },
            'comparison': {
                'show_both': True,  # Show both hollow and solid in same plot
                'interior_noise_levels': [0.05, 0.1, 0.2]  # Different noise levels to compare
            },
            'output': {
                'save_plots': False,
                'output_dir': 'results/plots',
                'formats': ['png', 'html'],
                'dpi': 300
            }
        }
    
    def generate_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate datasets for visualization."""
        datasets = {}
        
        # Dataset parameters
        n_samples = self.config['dataset']['n_samples']
        big_radius = self.config['dataset']['big_radius']
        small_radius = self.config['dataset']['small_radius']
        
        print(f"🎲 Generating datasets with {n_samples} points per torus...")
        print(f"   • Major radius: {big_radius}")
        print(f"   • Minor radius: {small_radius}")
        
        # Generate hollow torus
        print("   • Generating hollow torus...")
        X_hollow, y_hollow = generate(n_samples, big_radius, small_radius, solid=False)
        datasets['hollow'] = (X_hollow, y_hollow)
        
        # Generate solid tori with different noise levels
        noise_levels = self.config['comparison']['interior_noise_levels']
        for noise in noise_levels:
            print(f"   • Generating solid torus (noise={noise})...")
            X_solid, y_solid = generate(n_samples, big_radius, small_radius, 
                                      solid=True, interior_noise=noise)
            datasets[f'solid_noise_{noise}'] = (X_solid, y_solid)
        
        print(f"✅ Generated {len(datasets)} datasets")
        return datasets
    
    def create_plotly_visualization(self, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> go.Figure:
        """Create interactive 3D visualization using Plotly."""
        
        if self.config['comparison']['show_both']:
            # Create subplots for comparison
            n_datasets = len(datasets)
            cols = min(3, n_datasets)
            rows = (n_datasets + cols - 1) // cols
            
            subplot_titles = list(datasets.keys())
            fig = make_subplots(
                rows=rows, cols=cols,
                specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)],
                subplot_titles=subplot_titles,
                horizontal_spacing=0.05,
                vertical_spacing=0.1
            )
            
            colors = px.colors.qualitative.Set1
            
            for i, (name, (X, y)) in enumerate(datasets.items()):
                row = i // cols + 1
                col = i % cols + 1
                
                # Separate by class
                class_0_mask = (y.flatten() == 0)
                class_1_mask = (y.flatten() == 1)
                
                # Add class 0 points
                fig.add_trace(
                    go.Scatter3d(
                        x=X[class_0_mask, 0],
                        y=X[class_0_mask, 1], 
                        z=X[class_0_mask, 2],
                        mode='markers',
                        marker=dict(
                            size=self.config['visualization']['point_size'],
                            color=colors[0],
                            opacity=self.config['visualization']['alpha']
                        ),
                        name=f'{name} - Class 0',
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
                
                # Add class 1 points
                fig.add_trace(
                    go.Scatter3d(
                        x=X[class_1_mask, 0],
                        y=X[class_1_mask, 1],
                        z=X[class_1_mask, 2],
                        mode='markers',
                        marker=dict(
                            size=self.config['visualization']['point_size'],
                            color=colors[1],
                            opacity=self.config['visualization']['alpha']
                        ),
                        name=f'{name} - Class 1',
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
        else:
            # Single plot mode - use first dataset
            name, (X, y) = list(datasets.items())[0]
            fig = go.Figure()
            
            class_0_mask = (y.flatten() == 0)
            class_1_mask = (y.flatten() == 1)
            
            fig.add_trace(go.Scatter3d(
                x=X[class_0_mask, 0],
                y=X[class_0_mask, 1],
                z=X[class_0_mask, 2],
                mode='markers',
                marker=dict(
                    size=self.config['visualization']['point_size'],
                    color='red',
                    opacity=self.config['visualization']['alpha']
                ),
                name='Class 0'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=X[class_1_mask, 0],
                y=X[class_1_mask, 1],
                z=X[class_1_mask, 2],
                mode='markers',
                marker=dict(
                    size=self.config['visualization']['point_size'],
                    color='blue',
                    opacity=self.config['visualization']['alpha']
                ),
                name='Class 1'
            ))
        
        # Update layout
        fig.update_layout(
            title="🌍 Interactive 3D Torus Dataset Visualization",
            font=dict(size=12),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Z",
                xaxis=dict(showgrid=self.config['visualization']['show_grid']),
                yaxis=dict(showgrid=self.config['visualization']['show_grid']),
                zaxis=dict(showgrid=self.config['visualization']['show_grid']),
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=800
        )
        
        return fig
    
    def create_matplotlib_visualization(self, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> plt.Figure:
        """Create 3D visualization using matplotlib."""
        
        n_datasets = len(datasets)
        if self.config['comparison']['show_both'] and n_datasets > 1:
            cols = min(3, n_datasets)
            rows = (n_datasets + cols - 1) // cols
            
            fig = plt.figure(figsize=(self.config['visualization']['figure_size'][0] * cols,
                                    self.config['visualization']['figure_size'][1] * rows))
            
            for i, (name, (X, y)) in enumerate(datasets.items()):
                ax = fig.add_subplot(rows, cols, i+1, projection='3d')
                
                class_0_mask = (y.flatten() == 0)
                class_1_mask = (y.flatten() == 1)
                
                ax.scatter(X[class_0_mask, 0], X[class_0_mask, 1], X[class_0_mask, 2],
                          c='red', s=self.config['visualization']['point_size'], 
                          alpha=self.config['visualization']['alpha'], label='Class 0')
                ax.scatter(X[class_1_mask, 0], X[class_1_mask, 1], X[class_1_mask, 2],
                          c='blue', s=self.config['visualization']['point_size'],
                          alpha=self.config['visualization']['alpha'], label='Class 1')
                
                ax.set_title(f'Dataset: {name}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
                if i == 0:  # Only show legend for first subplot
                    ax.legend()
                
                if not self.config['visualization']['show_grid']:
                    ax.grid(False)
        else:
            # Single plot
            fig = plt.figure(figsize=self.config['visualization']['figure_size'])
            ax = fig.add_subplot(111, projection='3d')
            
            name, (X, y) = list(datasets.items())[0]
            
            class_0_mask = (y.flatten() == 0)
            class_1_mask = (y.flatten() == 1)
            
            ax.scatter(X[class_0_mask, 0], X[class_0_mask, 1], X[class_0_mask, 2],
                      c='red', s=self.config['visualization']['point_size'],
                      alpha=self.config['visualization']['alpha'], label='Class 0')
            ax.scatter(X[class_1_mask, 0], X[class_1_mask, 1], X[class_1_mask, 2],
                      c='blue', s=self.config['visualization']['point_size'],
                      alpha=self.config['visualization']['alpha'], label='Class 1')
            
            ax.set_title(f'🌍 3D Torus Dataset: {name}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y') 
            ax.set_zlabel('Z')
            ax.legend()
            
            if not self.config['visualization']['show_grid']:
                ax.grid(False)
        
        plt.tight_layout()
        return fig
    
    def save_plots(self, fig, backend: str):
        """Save plots to files."""
        if not self.config['output']['save_plots']:
            return
            
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        formats = self.config['output']['formats']
        
        for fmt in formats:
            if backend == 'plotly' and fmt == 'html':
                filepath = output_dir / f'torus_dataset_3d.html'
                fig.write_html(str(filepath))
                print(f"💾 Saved interactive plot: {filepath}")
            elif backend == 'plotly' and fmt == 'png':
                try:
                    filepath = output_dir / f'torus_dataset_3d.png'
                    fig.write_image(str(filepath), width=1200, height=800)
                    print(f"💾 Saved static plot: {filepath}")
                except Exception as e:
                    print(f"⚠️  Could not save PNG with plotly (install kaleido for image export): {e}")
                    print("   💡 HTML export will still work")
            elif backend == 'matplotlib' and fmt != 'html':
                filepath = output_dir / f'torus_dataset_3d.{fmt}'
                fig.savefig(str(filepath), dpi=self.config['output']['dpi'], 
                           bbox_inches='tight')
                print(f"💾 Saved plot: {filepath}")
            elif backend == 'matplotlib' and fmt == 'html':
                print(f"⚠️  HTML format not supported for matplotlib backend, skipping")
    
    def print_dataset_stats(self, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """Print statistical summary of datasets."""
        print("\n📊 DATASET STATISTICS")
        print("=" * 50)
        
        for name, (X, y) in datasets.items():
            print(f"\n🔍 Dataset: {name}")
            print(f"   • Shape: {X.shape}")
            print(f"   • Classes: {np.unique(y.flatten())}")
            print(f"   • Points per class: {np.bincount(y.astype(int).flatten())}")
            
            # Distance statistics
            center_distances = np.linalg.norm(X, axis=1)
            print(f"   • Distance from origin: {center_distances.mean():.3f} ± {center_distances.std():.3f}")
            print(f"   • Distance range: [{center_distances.min():.3f}, {center_distances.max():.3f}]")
            
            # Coordinate ranges
            print(f"   • X range: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
            print(f"   • Y range: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")
            print(f"   • Z range: [{X[:, 2].min():.3f}, {X[:, 2].max():.3f}]")
    
    def run_visualization(self, show_interactive: bool = True):
        """Run the complete visualization pipeline."""
        print("🎨 STARTING 3D TORUS DATASET VISUALIZATION")
        print("=" * 55)
        
        # Generate datasets
        datasets = self.generate_datasets()
        
        # Print statistics
        self.print_dataset_stats(datasets)
        
        # Create visualization
        backend = self.config['visualization']['backend']
        print(f"\n🖼️  Creating {backend} visualization...")
        
        if backend == 'plotly':
            fig = self.create_plotly_visualization(datasets)
            
            # Save plots
            self.save_plots(fig, 'plotly')
            
            # Show interactive plot
            if show_interactive:
                print("🌐 Opening interactive plot in browser...")
                fig.show()
            
        elif backend == 'matplotlib':
            fig = self.create_matplotlib_visualization(datasets)
            
            # Save plots  
            self.save_plots(fig, 'matplotlib')
            
            # Show plot
            if show_interactive:
                print("📊 Displaying matplotlib plot...")
                plt.show()
        
        print("\n✅ Visualization complete!")
        return fig


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Interactive 3D Torus Dataset Visualization")
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--interactive', action='store_true', 
                       help='Show interactive plots (default: True)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--backend', choices=['matplotlib', 'plotly'], 
                       default='plotly', help='Visualization backend')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TorusDataset3DVisualizer(args.config)
    
    # Override config with command line arguments
    if args.save_plots:
        visualizer.config['output']['save_plots'] = True
    if args.backend:
        visualizer.config['visualization']['backend'] = args.backend
    
    # Run visualization
    try:
        fig = visualizer.run_visualization(show_interactive=args.interactive)
        return fig
    except KeyboardInterrupt:
        print("\n⏹️  Visualization interrupted by user")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        raise


if __name__ == "__main__":
    main()