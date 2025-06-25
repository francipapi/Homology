"""
Decision Boundary Visualization

This module provides comprehensive 3D visualization capabilities for decision boundaries
extracted during neural network training. Features include:

- 3D isosurface rendering with interactive controls
- Training evolution animations
- Side-by-side architecture comparisons
- Topology overlay and analysis
- Export capabilities for presentations and papers

Author: Claude Code
Date: 2025
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import time
from dataclasses import dataclass

# Import for GIF/video creation
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    print("Warning: imageio not available. Animation export disabled.")
    IMAGEIO_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    print("Warning: trimesh not available. Advanced mesh operations disabled.")
    TRIMESH_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import boundary extraction results
from src.topology.decision_boundary_homology import BoundaryExtractionResult, load_boundary_config


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    backend: str = 'plotly'
    show_mesh: bool = True
    show_points: bool = True
    opacity: float = 0.7
    point_size: int = 2
    color_scheme: str = 'viridis'
    animation_fps: int = 2
    animation_duration: int = 10
    smooth_transitions: bool = True


class DecisionBoundaryVisualizer:
    """
    Comprehensive 3D visualization system for decision boundaries.
    
    This class provides methods for:
    1. 3D rendering of decision boundary isosurfaces
    2. Point cloud visualization
    3. Training evolution animations
    4. Architecture comparison plots
    5. Topology analysis overlays
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the visualizer.
        
        Parameters:
        - config: Visualization configuration dictionary
        """
        self.config = config or {}
        self.viz_config = self._parse_viz_config()
        
        # Storage for visualization data
        self.boundary_data = []  # List of BoundaryExtractionResult objects
        self.training_data = {}  # Training metrics
        self.color_schemes = self._setup_color_schemes()
        
    def _parse_viz_config(self) -> VisualizationConfig:
        """Parse visualization configuration."""
        viz_section = self.config.get('visualization', {})
        boundary_viz = viz_section.get('boundary_viz', {})
        animation = viz_section.get('animation', {})
        
        return VisualizationConfig(
            backend=boundary_viz.get('backend', 'plotly'),
            show_mesh=boundary_viz.get('show_mesh', True),
            show_points=boundary_viz.get('show_points', True),
            opacity=boundary_viz.get('opacity', 0.7),
            point_size=boundary_viz.get('point_size', 2),
            color_scheme=boundary_viz.get('color_scheme', 'viridis'),
            animation_fps=animation.get('fps', 2),
            animation_duration=animation.get('duration', 10),
            smooth_transitions=animation.get('smooth_transitions', True)
        )
    
    def _setup_color_schemes(self) -> Dict[str, Any]:
        """Setup color schemes for different visualization modes."""
        return {
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'inferno': px.colors.sequential.Inferno,
            'coolwarm': px.colors.diverging.RdBu,
            'topology': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
    
    def load_boundary_results(self, results_path: str) -> None:
        """
        Load boundary extraction results from file.
        
        Parameters:
        - results_path: Path to saved boundary results
        """
        try:
            if results_path.endswith('.pt'):
                data = torch.load(results_path, map_location='cpu')
                
                # Handle different data formats
                if 'boundary_results' in data:
                    # Direct format from training
                    self.boundary_data = data['boundary_results']
                    self.training_data = data.get('training_history', {})
                    print(f"Loaded {len(self.boundary_data)} boundary results")
                elif isinstance(data, dict) and len(data) > 0:
                    # Topology results format (architecture -> data)
                    first_arch = list(data.keys())[0]
                    arch_data = data[first_arch]
                    if 'boundary_results' in arch_data:
                        self.boundary_data = arch_data['boundary_results']
                        self.training_data = arch_data.get('training_history', {})
                        print(f"Loaded {len(self.boundary_data)} boundary results from {first_arch}")
                    else:
                        print("Warning: No boundary results found in architecture data")
                else:
                    print("Warning: No boundary results found in file")
            else:
                print(f"Unsupported file format: {results_path}")
        except Exception as e:
            print(f"Error loading boundary results: {e}")
    
    def load_boundary_data_from_directory(self, directory: str) -> None:
        """
        Load boundary data from individual files in a directory.
        
        Parameters:
        - directory: Directory containing boundary data files
        """
        try:
            boundary_dir = Path(directory)
            if not boundary_dir.exists():
                print(f"Directory not found: {directory}")
                return
            
            # Load topology files
            topology_files = sorted(boundary_dir.glob("topology_epoch_*.pt"))
            mesh_files = sorted(boundary_dir.glob("boundary_epoch_*.ply"))
            
            self.boundary_data = []
            
            for topo_file in topology_files:
                try:
                    topo_data = torch.load(topo_file, map_location='cpu')
                    
                    # Create BoundaryExtractionResult object
                    result = BoundaryExtractionResult(
                        epoch=topo_data['epoch'],
                        boundary_points=topo_data.get('boundary_points'),
                        betti_numbers=topo_data.get('betti_numbers'),
                        extraction_time=topo_data.get('extraction_time', 0),
                        topology_time=topo_data.get('topology_time', 0),
                        success=True,
                        metadata=topo_data.get('metadata')
                    )
                    
                    # Try to load corresponding mesh
                    epoch = topo_data['epoch']
                    mesh_file = boundary_dir / f"boundary_epoch_{epoch:04d}.ply"
                    if mesh_file.exists() and TRIMESH_AVAILABLE:
                        try:
                            mesh = trimesh.load(str(mesh_file))
                            result.mesh_vertices = mesh.vertices
                            result.mesh_faces = mesh.faces
                        except:
                            pass  # Continue without mesh if loading fails
                    
                    self.boundary_data.append(result)
                    
                except Exception as e:
                    print(f"Error loading {topo_file}: {e}")
            
            print(f"Loaded {len(self.boundary_data)} boundary results from directory")
            
        except Exception as e:
            print(f"Error loading from directory: {e}")
    
    def create_single_boundary_plot(self, result: BoundaryExtractionResult, 
                                   title: Optional[str] = None) -> go.Figure:
        """
        Create a 3D plot for a single decision boundary.
        
        Parameters:
        - result: BoundaryExtractionResult object
        - title: Plot title
        
        Returns:
        - fig: Plotly figure object
        """
        fig = go.Figure()
        
        # Add mesh if available
        if (self.viz_config.show_mesh and result.mesh_vertices is not None 
            and result.mesh_faces is not None):
            
            fig.add_trace(go.Mesh3d(
                x=result.mesh_vertices[:, 0],
                y=result.mesh_vertices[:, 1],
                z=result.mesh_vertices[:, 2],
                i=result.mesh_faces[:, 0],
                j=result.mesh_faces[:, 1],
                k=result.mesh_faces[:, 2],
                opacity=self.viz_config.opacity,
                colorscale=self.viz_config.color_scheme,
                name=f'Decision Boundary (Epoch {result.epoch})',
                showscale=False
            ))
        
        # Add point cloud if available
        if (self.viz_config.show_points and result.boundary_points is not None):
            
            # Color points by distance from origin (or topology if available)
            if result.betti_numbers is not None:
                # Color by topology complexity
                topology_complexity = sum(result.betti_numbers)
                colors = [topology_complexity] * len(result.boundary_points)
                colorbar_title = "Topology Complexity"
            else:
                # Color by distance from origin
                distances = np.linalg.norm(result.boundary_points, axis=1)
                colors = distances
                colorbar_title = "Distance from Origin"
            
            fig.add_trace(go.Scatter3d(
                x=result.boundary_points[:, 0],
                y=result.boundary_points[:, 1],
                z=result.boundary_points[:, 2],
                mode='markers',
                marker=dict(
                    size=self.viz_config.point_size,
                    color=colors,
                    colorscale=self.viz_config.color_scheme,
                    colorbar=dict(title=colorbar_title),
                    opacity=0.8
                ),
                name=f'Boundary Points (Epoch {result.epoch})'
            ))
        
        # Update layout
        plot_title = title or f'Decision Boundary - Epoch {result.epoch}'
        if result.betti_numbers is not None:
            betti_str = ', '.join([f'β{i}={b}' for i, b in enumerate(result.betti_numbers)])
            plot_title += f' | Betti: [{betti_str}]'
        
        fig.update_layout(
            title=plot_title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=600
        )
        
        return fig
    
    def create_evolution_animation(self, output_path: Optional[str] = None) -> go.Figure:
        """
        Create an animation showing decision boundary evolution during training.
        
        Parameters:
        - output_path: Path to save animation (optional)
        
        Returns:
        - fig: Plotly figure with animation
        """
        if not self.boundary_data:
            print("No boundary data available for animation")
            return go.Figure()
        
        # Sort by epoch
        sorted_data = sorted(self.boundary_data, key=lambda x: x.epoch)
        
        # Create animation frames
        frames = []
        
        for i, result in enumerate(sorted_data):
            frame_data = []
            
            # Add mesh trace if available
            if (self.viz_config.show_mesh and result.mesh_vertices is not None 
                and result.mesh_faces is not None):
                
                frame_data.append(go.Mesh3d(
                    x=result.mesh_vertices[:, 0],
                    y=result.mesh_vertices[:, 1],
                    z=result.mesh_vertices[:, 2],
                    i=result.mesh_faces[:, 0],
                    j=result.mesh_faces[:, 1],
                    k=result.mesh_faces[:, 2],
                    opacity=self.viz_config.opacity,
                    colorscale=self.viz_config.color_scheme,
                    name=f'Boundary Mesh',
                    showscale=False
                ))
            
            # Add point cloud trace if available
            if (self.viz_config.show_points and result.boundary_points is not None):
                
                distances = np.linalg.norm(result.boundary_points, axis=1)
                
                frame_data.append(go.Scatter3d(
                    x=result.boundary_points[:, 0],
                    y=result.boundary_points[:, 1],
                    z=result.boundary_points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=self.viz_config.point_size,
                        color=distances,
                        colorscale=self.viz_config.color_scheme,
                        opacity=0.8
                    ),
                    name='Boundary Points'
                ))
            
            # Create frame title
            frame_title = f'Epoch {result.epoch}'
            if result.betti_numbers is not None:
                betti_str = ', '.join([f'β{i}={b}' for i, b in enumerate(result.betti_numbers)])
                frame_title += f' | Betti: [{betti_str}]'
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(result.epoch),
                layout=go.Layout(title=f'Decision Boundary Evolution - {frame_title}')
            ))
        
        # Create initial figure with first frame
        fig = go.Figure(
            data=frames[0].data if frames else [],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title='Decision Boundary Evolution During Training',
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Z",
                aspectmode='cube'
            ),
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 1000 // self.viz_config.animation_fps, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300 if self.viz_config.smooth_transitions else 0}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 20},
                    'prefix': 'Epoch:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300 if self.viz_config.smooth_transitions else 0},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [
                    {
                        'args': [[result.epoch], {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }],
                        'label': str(result.epoch),
                        'method': 'animate'
                    }
                    for result in sorted_data
                ]
            }]
        )
        
        # Save animation if path provided
        if output_path:
            try:
                fig.write_html(output_path)
                print(f"Animation saved: {output_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
        
        return fig
    
    def create_topology_evolution_plot(self) -> go.Figure:
        """
        Create a plot showing how topology (Betti numbers) evolves during training.
        
        Returns:
        - fig: Plotly figure with topology evolution
        """
        if not self.boundary_data:
            print("No boundary data available for topology plot")
            return go.Figure()
        
        # Extract topology data
        epochs = []
        betti_data = {}
        
        for result in self.boundary_data:
            if result.betti_numbers is not None:
                epochs.append(result.epoch)
                
                for i, betti in enumerate(result.betti_numbers):
                    if i not in betti_data:
                        betti_data[i] = []
                    betti_data[i].append(betti)
        
        # Sort by epoch
        sorted_indices = np.argsort(epochs)
        epochs = [epochs[i] for i in sorted_indices]
        for dim in betti_data:
            betti_data[dim] = [betti_data[dim][i] for i in sorted_indices]
        
        # Create plot
        fig = go.Figure()
        
        # Betti number names
        betti_names = ['β₀ (Components)', 'β₁ (Loops)', 'β₂ (Voids)', 'β₃ (3-Cavities)']
        colors = self.color_schemes['topology']
        
        for dim, values in betti_data.items():
            name = betti_names[dim] if dim < len(betti_names) else f'β{dim}'
            color = colors[dim % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=epochs,
                y=values,
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Decision Boundary Topology Evolution',
            xaxis_title='Training Epoch',
            yaxis_title='Betti Number',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_comparison_plot(self, other_visualizer: 'DecisionBoundaryVisualizer',
                              labels: Tuple[str, str] = ('Architecture 1', 'Architecture 2')) -> go.Figure:
        """
        Create side-by-side comparison of decision boundaries from different architectures.
        
        Parameters:
        - other_visualizer: Another DecisionBoundaryVisualizer instance
        - labels: Labels for the two architectures
        
        Returns:
        - fig: Plotly figure with comparison
        """
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=labels,
            horizontal_spacing=0.05
        )
        
        # Get final boundary for each architecture
        if self.boundary_data:
            result1 = max(self.boundary_data, key=lambda x: x.epoch)
            self._add_boundary_to_subplot(fig, result1, row=1, col=1)
        
        if other_visualizer.boundary_data:
            result2 = max(other_visualizer.boundary_data, key=lambda x: x.epoch)
            self._add_boundary_to_subplot(fig, result2, row=1, col=2)
        
        fig.update_layout(
            title='Decision Boundary Comparison',
            height=600
        )
        
        return fig
    
    def _add_boundary_to_subplot(self, fig: go.Figure, result: BoundaryExtractionResult,
                                row: int, col: int) -> None:
        """Add boundary data to a subplot."""
        # Add mesh if available
        if (self.viz_config.show_mesh and result.mesh_vertices is not None 
            and result.mesh_faces is not None):
            
            fig.add_trace(go.Mesh3d(
                x=result.mesh_vertices[:, 0],
                y=result.mesh_vertices[:, 1],
                z=result.mesh_vertices[:, 2],
                i=result.mesh_faces[:, 0],
                j=result.mesh_faces[:, 1],
                k=result.mesh_faces[:, 2],
                opacity=self.viz_config.opacity,
                colorscale=self.viz_config.color_scheme,
                showscale=False,
                showlegend=False
            ), row=row, col=col)
        
        # Add points if available
        if (self.viz_config.show_points and result.boundary_points is not None):
            
            distances = np.linalg.norm(result.boundary_points, axis=1)
            
            fig.add_trace(go.Scatter3d(
                x=result.boundary_points[:, 0],
                y=result.boundary_points[:, 1],
                z=result.boundary_points[:, 2],
                mode='markers',
                marker=dict(
                    size=self.viz_config.point_size,
                    color=distances,
                    colorscale=self.viz_config.color_scheme,
                    opacity=0.8
                ),
                showlegend=False
            ), row=row, col=col)
    
    def create_topology_comparison_plot(self, other_visualizer: 'DecisionBoundaryVisualizer',
                                       labels: Tuple[str, str] = ('Architecture 1', 'Architecture 2')) -> go.Figure:
        """
        Create comparison plot of topology evolution between architectures.
        
        Parameters:
        - other_visualizer: Another DecisionBoundaryVisualizer instance
        - labels: Labels for the two architectures
        
        Returns:
        - fig: Plotly figure with topology comparison
        """
        fig = go.Figure()
        
        # Process first architecture
        epochs1, betti_data1 = self._extract_topology_data()
        self._add_topology_traces(fig, epochs1, betti_data1, labels[0], line_style='solid')
        
        # Process second architecture
        epochs2, betti_data2 = other_visualizer._extract_topology_data()
        other_visualizer._add_topology_traces(fig, epochs2, betti_data2, labels[1], line_style='dash')
        
        fig.update_layout(
            title='Topology Evolution Comparison',
            xaxis_title='Training Epoch',
            yaxis_title='Betti Number',
            hovermode='x unified'
        )
        
        return fig
    
    def _extract_topology_data(self) -> Tuple[List[int], Dict[int, List[int]]]:
        """Extract topology data from boundary results."""
        epochs = []
        betti_data = {}
        
        for result in self.boundary_data:
            if result.betti_numbers is not None:
                epochs.append(result.epoch)
                
                for i, betti in enumerate(result.betti_numbers):
                    if i not in betti_data:
                        betti_data[i] = []
                    betti_data[i].append(betti)
        
        # Sort by epoch
        sorted_indices = np.argsort(epochs)
        epochs = [epochs[i] for i in sorted_indices]
        for dim in betti_data:
            betti_data[dim] = [betti_data[dim][i] for i in sorted_indices]
        
        return epochs, betti_data
    
    def _add_topology_traces(self, fig: go.Figure, epochs: List[int], 
                           betti_data: Dict[int, List[int]], label: str, 
                           line_style: str = 'solid') -> None:
        """Add topology traces to figure."""
        betti_names = ['β₀', 'β₁', 'β₂', 'β₃']
        colors = self.color_schemes['topology']
        
        for dim, values in betti_data.items():
            name = f'{label} - {betti_names[dim] if dim < len(betti_names) else f"β{dim}"}'
            color = colors[dim % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=epochs,
                y=values,
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=3, dash=line_style),
                marker=dict(size=6)
            ))
    
    def export_visualization_data(self, output_path: str) -> None:
        """
        Export visualization data for external analysis.
        
        Parameters:
        - output_path: Path to save data
        """
        try:
            export_data = {
                'boundary_results': self.boundary_data,
                'training_data': self.training_data,
                'visualization_config': self.config
            }
            
            if output_path.endswith('.pt'):
                torch.save(export_data, output_path)
            elif output_path.endswith('.npz'):
                # Convert to numpy format
                np_data = {}
                for i, result in enumerate(self.boundary_data):
                    prefix = f'epoch_{result.epoch:04d}'
                    if result.boundary_points is not None:
                        np_data[f'{prefix}_points'] = result.boundary_points
                    if result.mesh_vertices is not None:
                        np_data[f'{prefix}_vertices'] = result.mesh_vertices
                        np_data[f'{prefix}_faces'] = result.mesh_faces
                    if result.betti_numbers is not None:
                        np_data[f'{prefix}_betti'] = np.array(result.betti_numbers)
                
                np.savez_compressed(output_path, **np_data)
            
            print(f"Visualization data exported: {output_path}")
            
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def show(self, fig: go.Figure) -> None:
        """Display a plotly figure."""
        fig.show()
    
    def save_plot(self, fig: go.Figure, output_path: str, **kwargs) -> None:
        """Save a plotly figure to file."""
        try:
            if output_path.endswith('.html'):
                fig.write_html(output_path, **kwargs)
            elif output_path.endswith('.png'):
                fig.write_image(output_path, **kwargs)
            elif output_path.endswith('.pdf'):
                fig.write_image(output_path, **kwargs)
            else:
                fig.write_html(output_path + '.html', **kwargs)
            
            print(f"Plot saved: {output_path}")
            
        except Exception as e:
            print(f"Error saving plot: {e}")


def load_and_visualize_boundaries(results_path: str, config_path: Optional[str] = None) -> DecisionBoundaryVisualizer:
    """
    Convenience function to load and visualize decision boundaries.
    
    Parameters:
    - results_path: Path to boundary results or directory
    - config_path: Path to visualization config (optional)
    
    Returns:
    - visualizer: Initialized DecisionBoundaryVisualizer
    """
    # Load config
    config = {}
    if config_path:
        config = load_boundary_config(config_path)
    
    # Create visualizer
    visualizer = DecisionBoundaryVisualizer(config)
    
    # Load data
    if Path(results_path).is_file():
        visualizer.load_boundary_results(results_path)
    else:
        visualizer.load_boundary_data_from_directory(results_path)
    
    return visualizer


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Decision Boundary Visualization")
    parser.add_argument('results_path', type=str, help='Path to boundary results file or directory')
    parser.add_argument('--config', type=str, help='Path to visualization config file')
    parser.add_argument('--output', type=str, help='Output path for saved plots')
    parser.add_argument('--animation', action='store_true', help='Create evolution animation')
    parser.add_argument('--topology', action='store_true', help='Create topology evolution plot')
    parser.add_argument('--export-data', type=str, help='Export visualization data')
    
    args = parser.parse_args()
    
    try:
        # Load and create visualizer
        visualizer = load_and_visualize_boundaries(args.results_path, args.config)
        
        if not visualizer.boundary_data:
            print("No boundary data loaded. Exiting.")
            return
        
        print(f"Loaded {len(visualizer.boundary_data)} boundary results")
        
        # Create visualizations
        if args.animation:
            print("Creating evolution animation...")
            fig = visualizer.create_evolution_animation()
            if args.output:
                output_path = args.output if args.output.endswith('.html') else args.output + '_animation.html'
                visualizer.save_plot(fig, output_path)
            else:
                visualizer.show(fig)
        
        elif args.topology:
            print("Creating topology evolution plot...")
            fig = visualizer.create_topology_evolution_plot()
            if args.output:
                output_path = args.output if args.output.endswith('.html') else args.output + '_topology.html'
                visualizer.save_plot(fig, output_path)
            else:
                visualizer.show(fig)
        
        else:
            # Create single boundary plot (final epoch)
            print("Creating single boundary plot...")
            final_result = max(visualizer.boundary_data, key=lambda x: x.epoch)
            fig = visualizer.create_single_boundary_plot(final_result)
            if args.output:
                output_path = args.output if args.output.endswith('.html') else args.output + '_boundary.html'
                visualizer.save_plot(fig, output_path)
            else:
                visualizer.show(fig)
        
        # Export data if requested
        if args.export_data:
            visualizer.export_visualization_data(args.export_data)
        
        print("Visualization completed!")
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        raise


if __name__ == "__main__":
    main()