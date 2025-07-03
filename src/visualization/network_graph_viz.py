"""
Network Graph Visualization Module

This module provides visualization tools for neural network graphs created by
the NetworkGraphBuilder. It supports both static and interactive visualizations,
with special handling for factor graph representations of convolutional layers.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import networkx as nx
import graph_tool as gt
from graph_tool import Graph
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.network_graph_builder import UnifiedGraphBuilder


class NetworkGraphVisualizer:
    """
    Visualizer for neural network graphs with support for different layer types.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        # Color schemes for different node types
        self.node_colors = {
            'input': '#90EE90',      # Light green
            'hidden': '#87CEEB',     # Sky blue
            'output': '#FFB6C1',     # Light pink
            'parameter': '#FFD700',  # Gold
            'bias': '#DDA0DD',       # Plum
        }
        
        # Node sizes
        self.node_sizes = {
            'input': 30,
            'hidden': 25,
            'output': 30,
            'parameter': 15,
            'bias': 20,
        }
        
    def visualize_network(self, model: nn.Module, 
                         method: str = 'interactive',
                         save_path: Optional[str] = None,
                         show_weights: bool = False,
                         layout: str = 'hierarchical',
                         figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize a neural network graph.
        
        Args:
            model: PyTorch model to visualize
            method: Visualization method ('static', 'interactive', 'both')
            save_path: Path to save the visualization
            show_weights: Whether to show edge weights
            layout: Graph layout algorithm
            figsize: Figure size for static plots
        """
        # Build network graph
        builder = UnifiedGraphBuilder()
        graph = builder.build_network_graph(model)
        
        print(f"Network graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
        
        if method in ['static', 'both']:
            self._visualize_static(graph, model, save_path, show_weights, layout, figsize)
            
        if method in ['interactive', 'both']:
            interactive_path = save_path.replace('.png', '.html') if save_path else None
            self._visualize_interactive(graph, model, interactive_path, show_weights, layout)
    
    def _visualize_static(self, graph: Graph, model: nn.Module,
                         save_path: Optional[str] = None,
                         show_weights: bool = False,
                         layout: str = 'hierarchical',
                         figsize: Tuple[int, int] = (12, 8)) -> None:
        """Create static visualization using matplotlib."""
        # Convert to NetworkX for easier visualization
        nx_graph = self._graph_tool_to_networkx(graph)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate layout
        pos = self._calculate_layout(nx_graph, layout)
        
        # Draw nodes by type
        for node_type, color in self.node_colors.items():
            nodes = [n for n, d in nx_graph.nodes(data=True) 
                    if d.get('type') == node_type]
            if nodes:
                nx.draw_networkx_nodes(
                    nx_graph, pos, nodelist=nodes,
                    node_color=color,
                    node_size=self.node_sizes.get(node_type, 25) * 10,
                    label=node_type.capitalize(),
                    ax=ax
                )
        
        # Draw edges
        edge_widths = []
        edge_colors = []
        for u, v, d in nx_graph.edges(data=True):
            weight = d.get('weight', 1.0)
            edge_widths.append(min(abs(weight) * 2, 5))
            edge_colors.append('red' if weight < 0 else 'gray')
        
        nx.draw_networkx_edges(
            nx_graph, pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            ax=ax
        )
        
        # Draw labels for small graphs
        if graph.num_vertices() < 50:
            labels = {n: f"{d.get('neuron_idx', n)}" 
                     for n, d in nx_graph.nodes(data=True)
                     if d.get('type') != 'parameter'}
            nx.draw_networkx_labels(nx_graph, pos, labels, font_size=8, ax=ax)
        
        # Add title and legend
        ax.set_title(f"Network Architecture Graph\n{type(model).__name__}", fontsize=14)
        ax.legend(loc='best')
        ax.axis('off')
        
        # Add text info
        info_text = f"Vertices: {graph.num_vertices()}\nEdges: {graph.num_edges()}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Static visualization saved to {save_path}")
        else:
            plt.show()
    
    def _visualize_interactive(self, graph: Graph, model: nn.Module,
                             save_path: Optional[str] = None,
                             show_weights: bool = False,
                             layout: str = 'hierarchical') -> None:
        """Create interactive visualization using Plotly."""
        # Convert to NetworkX
        nx_graph = self._graph_tool_to_networkx(graph)
        
        # Calculate layout
        pos = self._calculate_layout(nx_graph, layout)
        
        # Create edge traces
        edge_traces = []
        
        for u, v, d in nx_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            weight = d.get('weight', 1.0)
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=min(abs(weight) * 2, 5),
                    color='red' if weight < 0 else 'gray'
                ),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node traces by type
        node_traces = []
        
        for node_type, color in self.node_colors.items():
            nodes = [(n, d) for n, d in nx_graph.nodes(data=True) 
                    if d.get('type') == node_type]
            
            if not nodes:
                continue
                
            node_x = []
            node_y = []
            node_text = []
            
            for node, data in nodes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Create hover text
                text = f"Type: {node_type}<br>"
                text += f"Layer: {data.get('layer', 'N/A')}<br>"
                text += f"Index: {data.get('neuron_idx', 'N/A')}"
                
                if node_type == 'parameter':
                    # Add weight info for parameter nodes
                    weight = self._get_parameter_weight(graph, node)
                    if weight is not None:
                        text += f"<br>Weight: {weight:.4f}"
                
                node_text.append(text)
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                marker=dict(
                    size=self.node_sizes.get(node_type, 25),
                    color=color,
                    line=dict(width=2, color='black')
                ),
                text=node_text,
                hoverinfo='text',
                name=node_type.capitalize()
            )
            node_traces.append(node_trace)
        
        # Create figure
        fig = go.Figure(data=edge_traces + node_traces)
        
        # Update layout
        fig.update_layout(
            title=f"Interactive Network Graph - {type(model).__name__}",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Add annotations for layer boundaries
        if layout == 'hierarchical':
            self._add_layer_annotations(fig, nx_graph, pos)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive visualization saved to {save_path}")
        else:
            fig.show()
    
    def visualize_factor_graph_detail(self, model: nn.Module, layer_name: str,
                                    save_path: Optional[str] = None) -> None:
        """
        Visualize detailed factor graph for a specific convolutional layer.
        
        Args:
            model: PyTorch model
            layer_name: Name of the convolutional layer to visualize
            save_path: Path to save visualization
        """
        # Find the layer
        layer = None
        for name, module in model.named_modules():
            if name == layer_name and isinstance(module, (nn.Conv1d, nn.Conv2d)):
                layer = module
                break
        
        if layer is None:
            raise ValueError(f"Convolutional layer '{layer_name}' not found")
        
        # Build factor graph for this layer
        from src.utils.network_graph_builder import ConvGraphBuilder
        builder = ConvGraphBuilder()
        
        # Determine input shape (simplified)
        if isinstance(layer, nn.Conv2d):
            input_shape = (layer.in_channels, 32, 32)  # Example shape
        else:
            input_shape = (layer.in_channels, 32)
        
        graph, _ = builder.build_graph(layer, input_shape)
        
        # Create detailed visualization
        self._visualize_factor_graph_matplotlib(graph, layer, save_path)
    
    def _visualize_factor_graph_matplotlib(self, graph: Graph, layer: nn.Module,
                                         save_path: Optional[str] = None) -> None:
        """Create detailed factor graph visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Simplified view
        self._draw_factor_graph_simplified(ax1, graph, layer)
        
        # Right: Weight distribution
        self._draw_weight_distribution(ax2, graph, layer)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Factor graph visualization saved to {save_path}")
        else:
            plt.show()
    
    def _draw_factor_graph_simplified(self, ax, graph: Graph, layer: nn.Module):
        """Draw simplified factor graph representation."""
        # Count node types
        v_type = graph.vp.type
        node_counts = {}
        for v in graph.vertices():
            t = v_type[v]
            node_counts[t] = node_counts.get(t, 0) + 1
        
        # Create simplified representation
        y_positions = {'input': 0.8, 'parameter': 0.5, 'output': 0.2}
        x_spacing = 0.8 / max(3, max(node_counts.values()) // 100)
        
        # Draw node groups
        for node_type, y in y_positions.items():
            count = node_counts.get(node_type, 0)
            if count == 0:
                continue
            
            # Draw representative nodes
            num_repr = min(5, count)
            x_start = 0.1
            
            for i in range(num_repr):
                x = x_start + i * x_spacing
                circle = Circle((x, y), 0.02, 
                              color=self.node_colors[node_type],
                              ec='black', linewidth=2)
                ax.add_patch(circle)
            
            # Add count label
            ax.text(x_start + num_repr * x_spacing + 0.05, y,
                   f"{count} nodes", va='center', fontsize=10)
        
        # Draw connections
        ax.arrow(0.3, 0.75, 0, -0.2, head_width=0.02, head_length=0.02,
                fc='gray', ec='gray')
        ax.arrow(0.3, 0.45, 0, -0.2, head_width=0.02, head_length=0.02,
                fc='gray', ec='gray')
        
        # Add labels
        ax.text(0.05, 0.85, "Input\nActivations", fontsize=12, weight='bold')
        ax.text(0.05, 0.5, "Parameter\nNodes", fontsize=12, weight='bold')
        ax.text(0.05, 0.15, "Output\nActivations", fontsize=12, weight='bold')
        
        # Add title
        ax.set_title(f"Factor Graph Structure\n{type(layer).__name__}", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _draw_weight_distribution(self, ax, graph: Graph, layer: nn.Module):
        """Draw weight distribution histogram."""
        # Extract weights from self-loops
        weights = []
        e_weight = graph.ep.weight
        v_type = graph.vp.type
        
        for v in graph.vertices():
            if v_type[v] == 'parameter':
                for e in v.out_edges():
                    if e.target() == v:  # Self-loop
                        weights.append(e_weight[e])
        
        if weights:
            ax.hist(weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(weights), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(weights):.3f}')
            ax.axvline(np.median(weights), color='green', linestyle='--',
                      label=f'Median: {np.median(weights):.3f}')
            
            ax.set_xlabel('Weight Magnitude')
            ax.set_ylabel('Count')
            ax.set_title(f'Weight Distribution\n{len(weights)} parameters')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No weights found', ha='center', va='center')
            ax.set_title('Weight Distribution')
    
    def _graph_tool_to_networkx(self, gt_graph: Graph) -> nx.DiGraph:
        """Convert graph-tool graph to NetworkX format."""
        nx_graph = nx.DiGraph()
        
        # Copy vertices with properties
        v_type = gt_graph.vp.type
        v_layer = gt_graph.vp.layer
        v_neuron_idx = gt_graph.vp.neuron_idx
        
        for v in gt_graph.vertices():
            nx_graph.add_node(
                int(v),
                type=v_type[v],
                layer=v_layer[v],
                neuron_idx=v_neuron_idx[v]
            )
        
        # Copy edges with weights
        e_weight = gt_graph.ep.weight
        for e in gt_graph.edges():
            nx_graph.add_edge(
                int(e.source()),
                int(e.target()),
                weight=e_weight[e]
            )
        
        return nx_graph
    
    def _calculate_layout(self, nx_graph: nx.DiGraph, layout: str) -> Dict:
        """Calculate node positions for visualization."""
        if layout == 'hierarchical':
            # Group nodes by layer
            layers = {}
            for node, data in nx_graph.nodes(data=True):
                layer = data.get('layer', 0)
                if layer not in layers:
                    layers[layer] = []
                layers[layer].append(node)
            
            # Calculate positions
            pos = {}
            layer_keys = sorted(layers.keys())
            x_spacing = 2.0 / max(1, len(layer_keys) - 1) if len(layer_keys) > 1 else 0
            
            for i, layer in enumerate(layer_keys):
                x = i * x_spacing
                nodes = layers[layer]
                y_spacing = 2.0 / max(1, len(nodes) - 1) if len(nodes) > 1 else 0
                
                for j, node in enumerate(nodes):
                    y = j * y_spacing - 1.0
                    pos[node] = (x, y)
        
        elif layout == 'spring':
            pos = nx.spring_layout(nx_graph, k=1/np.sqrt(nx_graph.number_of_nodes()))
        
        elif layout == 'circular':
            pos = nx.circular_layout(nx_graph)
        
        else:
            pos = nx.random_layout(nx_graph)
        
        return pos
    
    def _add_layer_annotations(self, fig, nx_graph, pos):
        """Add layer boundary annotations to interactive plot."""
        # Group nodes by layer
        layers = {}
        for node, data in nx_graph.nodes(data=True):
            layer = data.get('layer', 0)
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(pos[node][0])  # x-coordinate
        
        # Add vertical lines between layers
        for layer in sorted(layers.keys())[:-1]:
            if layer + 1 in layers:
                x1 = max(layers[layer])
                x2 = min(layers[layer + 1])
                x_mid = (x1 + x2) / 2
                
                fig.add_shape(
                    type="line",
                    x0=x_mid, y0=-2, x1=x_mid, y1=2,
                    line=dict(color="lightgray", width=2, dash="dash")
                )
    
    def _get_parameter_weight(self, graph: Graph, node: int) -> Optional[float]:
        """Get weight value from parameter node's self-loop."""
        e_weight = graph.ep.weight
        v = graph.vertex(node)
        
        for e in v.out_edges():
            if e.target() == v:  # Self-loop
                return float(e_weight[e])
        
        return None
    
    def create_animation(self, model_states: List[Dict[str, Any]], 
                        output_path: str,
                        fps: int = 2) -> None:
        """
        Create animation showing network evolution during training.
        
        Args:
            model_states: List of model state dicts at different training steps
            output_path: Path to save animation
            fps: Frames per second
        """
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def update(frame):
            ax.clear()
            
            # Create model from state dict
            model = self._create_model_from_state(model_states[frame])
            
            # Build and visualize graph
            builder = UnifiedGraphBuilder()
            graph = builder.build_network_graph(model)
            
            # Simple visualization for animation
            nx_graph = self._graph_tool_to_networkx(graph)
            pos = self._calculate_layout(nx_graph, 'hierarchical')
            
            # Draw
            for node_type, color in self.node_colors.items():
                nodes = [n for n, d in nx_graph.nodes(data=True) 
                        if d.get('type') == node_type]
                if nodes:
                    nx.draw_networkx_nodes(
                        nx_graph, pos, nodelist=nodes,
                        node_color=color,
                        node_size=100,
                        ax=ax
                    )
            
            nx.draw_networkx_edges(nx_graph, pos, alpha=0.3, ax=ax)
            
            ax.set_title(f"Network Evolution - Step {frame}")
            ax.axis('off')
        
        anim = animation.FuncAnimation(
            fig, update, frames=len(model_states),
            interval=1000/fps, blit=False
        )
        
        anim.save(output_path, writer='pillow')
        print(f"Animation saved to {output_path}")


def visualize_network_from_file(model_path: str, 
                              visualization_type: str = 'interactive',
                              save_path: Optional[str] = None) -> None:
    """
    Convenience function to visualize a saved model.
    
    Args:
        model_path: Path to saved model
        visualization_type: Type of visualization
        save_path: Path to save visualization
    """
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model (you need to know the architecture)
    # This is a simplified example
    from src.models.torch_mlp import MLP
    model = MLP(3, 8, 32, 1)  # Example architecture
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Visualize
    visualizer = NetworkGraphVisualizer()
    visualizer.visualize_network(model, visualization_type, save_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize neural network graphs")
    parser.add_argument("--model", type=str, help="Path to saved model")
    parser.add_argument("--type", type=str, default="interactive",
                       choices=["static", "interactive", "both"],
                       help="Visualization type")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--layout", type=str, default="hierarchical",
                       choices=["hierarchical", "spring", "circular"],
                       help="Graph layout algorithm")
    
    args = parser.parse_args()
    
    if args.model:
        visualize_network_from_file(args.model, args.type, args.output)
    else:
        # Demo with a simple network
        import torch.nn as nn
        
        model = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
        model.input_shape = (3,)
        
        visualizer = NetworkGraphVisualizer()
        visualizer.visualize_network(model, args.type, args.output)