#!/usr/bin/env python3
"""
Script to visualize neural network graphs with various options.

Examples:
    # Visualize a simple MLP
    python visualize_network_graph.py --model-type mlp
    
    # Visualize a CNN with factor graphs
    python visualize_network_graph.py --model-type cnn
    
    # Load and visualize a saved model
    python visualize_network_graph.py --load-model path/to/model.pt
    
    # Save visualization
    python visualize_network_graph.py --model-type mlp --output network_graph.html
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path

# Add project to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.visualization.network_graph_viz import NetworkGraphVisualizer
from src.models.torch_mlp import MLP
from src.models.torch_custom import CustomNet


def create_demo_mlp():
    """Create a demo MLP for visualization."""
    model = nn.Sequential(
        nn.Linear(3, 10),
        nn.ReLU(),
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )
    model.input_shape = (3,)
    return model


def create_demo_cnn():
    """Create a demo CNN for visualization."""
    class DemoCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 7 * 7, 32)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(32, 10)
            self.input_shape = (1, 28, 28)
            
        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x
    
    return DemoCNN()


def create_custom_model():
    """Create a model using the custom architecture from config."""
    import yaml
    
    config_path = Path("configs/training_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config.get('custom_architecture', {}).get('enabled', False):
            return CustomNet(config['custom_architecture'])
    
    # Fallback to default custom architecture
    custom_config = {
        'input_shape': [3],
        'layers': [
            {'type': 'linear', 'out_features': 16, 'activation': 'relu'},
            {'type': 'reshape', 'shape': [4, 4]},
            {'type': 'conv1d', 'out_channels': 8, 'kernel_size': 3, 
             'padding': 1, 'activation': 'relu'},
            {'type': 'flatten'},
            {'type': 'linear', 'out_features': 1, 'activation': 'sigmoid'}
        ]
    }
    return CustomNet(custom_config)


def load_model_from_checkpoint(checkpoint_path: str):
    """Load a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to infer model type from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        if config.get('custom_architecture', {}).get('enabled', False):
            model = CustomNet(config['custom_architecture'])
        else:
            # Default MLP
            model_config = config['model']
            model = MLP(
                input_dim=model_config['input_dim'],
                num_hidden_layers=model_config['num_hidden_layers'],
                hidden_dim=model_config['hidden_dim'],
                output_dim=model_config['output_dim'],
                activation_fn_name=model_config['activation_fn_name']
            )
            model.input_shape = (model_config['input_dim'],)
    else:
        raise ValueError("Cannot infer model architecture from checkpoint")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main():
    parser = argparse.ArgumentParser(description="Visualize neural network graphs")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-type", type=str, 
                           choices=["mlp", "cnn", "custom"],
                           help="Type of demo model to create")
    model_group.add_argument("--load-model", type=str,
                           help="Path to saved model checkpoint")
    
    # Visualization options
    parser.add_argument("--viz-type", type=str, default="interactive",
                       choices=["static", "interactive", "both"],
                       help="Visualization type")
    parser.add_argument("--layout", type=str, default="hierarchical",
                       choices=["hierarchical", "spring", "circular"],
                       help="Graph layout algorithm")
    parser.add_argument("--show-weights", action="store_true",
                       help="Show edge weights in visualization")
    parser.add_argument("--output", type=str,
                       help="Output path for saving visualization")
    parser.add_argument("--figsize", type=int, nargs=2, default=[12, 8],
                       help="Figure size for static plots")
    
    # Factor graph detail
    parser.add_argument("--factor-detail", type=str,
                       help="Show detailed factor graph for specific conv layer")
    
    args = parser.parse_args()
    
    # Create or load model
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = load_model_from_checkpoint(args.load_model)
    else:
        print(f"Creating demo {args.model_type.upper()} model")
        if args.model_type == "mlp":
            model = create_demo_mlp()
        elif args.model_type == "cnn":
            model = create_demo_cnn()
        elif args.model_type == "custom":
            model = create_custom_model()
    
    # Create visualizer
    visualizer = NetworkGraphVisualizer()
    
    # Show model info
    print(f"\nModel architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Visualize
    if args.factor_detail and args.model_type == "cnn":
        # Show detailed factor graph for specific layer
        print(f"\nVisualizing factor graph for layer: {args.factor_detail}")
        output_path = args.output or f"factor_graph_{args.factor_detail}.png"
        visualizer.visualize_factor_graph_detail(model, args.factor_detail, output_path)
    else:
        # Full network visualization
        print(f"\nCreating {args.viz_type} visualization...")
        visualizer.visualize_network(
            model,
            method=args.viz_type,
            save_path=args.output,
            show_weights=args.show_weights,
            layout=args.layout,
            figsize=tuple(args.figsize)
        )
    
    if args.output:
        print(f"\nVisualization saved to: {args.output}")
    
    # Print graph statistics
    from src.utils.network_graph_builder import UnifiedGraphBuilder
    builder = UnifiedGraphBuilder()
    graph = builder.build_network_graph(model)
    
    print(f"\nGraph statistics:")
    print(f"- Vertices: {graph.num_vertices()}")
    print(f"- Edges: {graph.num_edges()}")
    
    # Count node types
    v_type = graph.vp.type
    node_types = {}
    for v in graph.vertices():
        t = v_type[v]
        node_types[t] = node_types.get(t, 0) + 1
    
    print("\nNode types:")
    for node_type, count in sorted(node_types.items()):
        print(f"- {node_type}: {count}")


if __name__ == "__main__":
    main()