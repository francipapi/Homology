#!/usr/bin/env python3
"""
Debug script to compare weight distributions and edge normalization 
between trained CNNs and random CNNs.
"""

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.topology.network_homology_tracker import NetworkHomologyTracker
from src.models.torch_custom import CustomNet


def load_cnn_model(model_path: Path):
    """Load a CNN model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', {})
    
    model = CustomNet(config['custom_architecture'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    if not hasattr(model, 'input_shape'):
        input_dim = config.get('model', {}).get('input_dim', 3)
        model.input_shape = (input_dim,)
    
    return model, config


def create_random_cnn(reference_config):
    """Create a random CNN with same architecture as reference."""
    model = CustomNet(reference_config['custom_architecture'])
    
    # Use same initialization as in the test
    torch.manual_seed(999999)
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    model.eval()
    
    if not hasattr(model, 'input_shape'):
        input_dim = reference_config.get('model', {}).get('input_dim', 3)
        model.input_shape = (input_dim,)
    
    return model


def analyze_model_weights(model, model_name):
    """Analyze weight distributions in the model."""
    print(f"\n=== {model_name} Weight Analysis ===")
    
    all_weights = []
    layer_info = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            weights = module.weight.detach().cpu().numpy().flatten()
            all_weights.extend(weights)
            
            layer_info.append({
                'name': name,
                'type': type(module).__name__,
                'weights': weights,
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'abs_max': np.max(np.abs(weights))
            })
    
    all_weights = np.array(all_weights)
    
    print(f"Overall weight statistics:")
    print(f"  Mean: {np.mean(all_weights):.6f}")
    print(f"  Std:  {np.std(all_weights):.6f}")
    print(f"  Min:  {np.min(all_weights):.6f}")
    print(f"  Max:  {np.max(all_weights):.6f}")
    print(f"  Abs Max: {np.max(np.abs(all_weights)):.6f}")
    
    print(f"\nPer-layer analysis:")
    for info in layer_info:
        print(f"  {info['name']} ({info['type']}):")
        print(f"    Range: [{info['min']:.4f}, {info['max']:.4f}]")
        print(f"    Mean±Std: {info['mean']:.4f}±{info['std']:.4f}")
        print(f"    Abs Max: {info['abs_max']:.4f}")
    
    return all_weights, layer_info


def analyze_graph_edge_weights(model, tracker, model_name):
    """Analyze edge weights in the constructed graph."""
    print(f"\n=== {model_name} Graph Edge Analysis ===")
    
    # Build graph
    graph = tracker.graph_builder.build_network_graph(model)
    print(f"Graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
    
    # Extract edge weights
    edge_weights = []
    for edge in graph.edges():
        weight = graph.ep.weight[edge]
        edge_weights.append(weight)
    
    edge_weights = np.array(edge_weights)
    
    print(f"Edge weight statistics:")
    print(f"  Mean: {np.mean(edge_weights):.6f}")
    print(f"  Std:  {np.std(edge_weights):.6f}")
    print(f"  Min:  {np.min(edge_weights):.6f}")
    print(f"  Max:  {np.max(edge_weights):.6f}")
    print(f"  Median: {np.median(edge_weights):.6f}")
    
    # Count edges in different ranges
    ranges = [
        (0.0, 0.1),
        (0.1, 0.5),
        (0.5, 0.8),
        (0.8, 1.0),
        (1.0, float('inf'))
    ]
    
    print(f"\nEdge weight distribution:")
    for low, high in ranges:
        if high == float('inf'):
            count = np.sum(edge_weights >= low)
            print(f"  [{low:.1f}, ∞): {count:,} edges ({100*count/len(edge_weights):.1f}%)")
        else:
            count = np.sum((edge_weights >= low) & (edge_weights < high))
            print(f"  [{low:.1f}, {high:.1f}): {count:,} edges ({100*count/len(edge_weights):.1f}%)")
    
    return edge_weights, graph


def test_dimension_2_computation(model, tracker, model_name):
    """Test if dimension 2 computation works for this model."""
    print(f"\n=== Testing Dimension 2 Computation for {model_name} ===")
    
    try:
        # Try with dimension 2
        config_dim2 = tracker.config.copy()
        config_dim2['network_homology']['simplicial_complex']['max_dimension'] = 2
        config_dim2['network_homology']['persistence']['epsilon_filtering'] = None
        
        tracker_dim2 = NetworkHomologyTracker(config_dim2)
        
        print("Attempting dimension 2 computation...")
        _, snapshot = tracker_dim2.track_training_step(model=model, step=0, epoch=0)
        
        total_points = sum(len(dgm) for dgm in snapshot.persistence_diagrams.values())
        print(f"✅ SUCCESS: {total_points} total persistence points")
        
        for dim, dgm in snapshot.persistence_diagrams.items():
            print(f"  Dimension {dim}: {len(dgm)} points")
        
        return True, snapshot.persistence_diagrams
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False, None


def main():
    print("=" * 80)
    print("Weight Normalization and Topological Complexity Analysis")
    print("=" * 80)
    
    # Load configuration
    config_path = "configs/network_homology_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['network_homology']['persistence']['epsilon_filtering'] = None
    config['network_homology']['simplicial_complex']['max_dimension'] = 1  # Start with 1
    
    tracker = NetworkHomologyTracker(config)
    
    print(f"\nGraph builder configuration:")
    print(f"  Normalize weights: {tracker.graph_builder.normalize_weights}")
    print(f"  Weight threshold: {tracker.graph_builder.weight_threshold}")
    print(f"  Weight encoding: {tracker.graph_builder.weight_encoding}")
    print(f"  Normalization type: {tracker.graph_builder.normalization_type}")
    
    # Load trained CNN
    model_path = Path("results/homology_training/custom_1751557522/model.pt")
    if not model_path.exists():
        print("Trained model not found!")
        return
    
    print("\n" + "="*50)
    print("1. TRAINED CNN ANALYSIS")
    print("="*50)
    
    trained_model, model_config = load_cnn_model(model_path)
    
    # Analyze weights
    trained_weights, trained_layer_info = analyze_model_weights(trained_model, "Trained CNN")
    
    # Analyze graph edge weights
    trained_edge_weights, trained_graph = analyze_graph_edge_weights(trained_model, tracker, "Trained CNN")
    
    # Test dimension 2
    trained_dim2_success, trained_dim2_diagrams = test_dimension_2_computation(trained_model, tracker, "Trained CNN")
    
    print("\n" + "="*50)
    print("2. RANDOM CNN ANALYSIS")
    print("="*50)
    
    # Create random CNN
    random_model = create_random_cnn(model_config)
    
    # Analyze weights
    random_weights, random_layer_info = analyze_model_weights(random_model, "Random CNN")
    
    # Analyze graph edge weights
    random_edge_weights, random_graph = analyze_graph_edge_weights(random_model, tracker, "Random CNN")
    
    # Test dimension 2
    random_dim2_success, random_dim2_diagrams = test_dimension_2_computation(random_model, tracker, "Random CNN")
    
    print("\n" + "="*50)
    print("3. COMPARISON AND ANALYSIS")
    print("="*50)
    
    print(f"\nRaw weight comparison:")
    print(f"  Trained weights range: [{np.min(trained_weights):.4f}, {np.max(trained_weights):.4f}]")
    print(f"  Random weights range:  [{np.min(random_weights):.4f}, {np.max(random_weights):.4f}]")
    print(f"  Trained abs max: {np.max(np.abs(trained_weights)):.4f}")
    print(f"  Random abs max:  {np.max(np.abs(random_weights)):.4f}")
    
    print(f"\nGraph edge weight comparison:")
    print(f"  Trained edges range: [{np.min(trained_edge_weights):.4f}, {np.max(trained_edge_weights):.4f}]")
    print(f"  Random edges range:  [{np.min(random_edge_weights):.4f}, {np.max(random_edge_weights):.4f}]")
    
    print(f"\nDimension 2 computation:")
    print(f"  Trained CNN: {'✅ SUCCESS' if trained_dim2_success else '❌ FAILED'}")
    print(f"  Random CNN:  {'✅ SUCCESS' if random_dim2_success else '❌ FAILED'}")
    
    # If one fails and other succeeds, analyze why
    if trained_dim2_success != random_dim2_success:
        print(f"\n🔍 ANALYSIS: Different dimension 2 behavior detected!")
        
        if trained_dim2_success and not random_dim2_success:
            print(f"  - Trained CNN works with dim 2, but random CNN fails")
            print(f"  - This suggests random weights create more complex topology")
            print(f"  - Random edge weights may be less normalized/more spread out")
            
        # Check edge weight distributions
        print(f"\nEdge weight distribution analysis:")
        print(f"  Trained edges > 0.9: {np.sum(trained_edge_weights > 0.9):,} ({100*np.sum(trained_edge_weights > 0.9)/len(trained_edge_weights):.1f}%)")
        print(f"  Random edges > 0.9:  {np.sum(random_edge_weights > 0.9):,} ({100*np.sum(random_edge_weights > 0.9)/len(random_edge_weights):.1f}%)")
        
        print(f"\n💡 Hypothesis:")
        print(f"  - Random weights may create more complex simplicial structure")
        print(f"  - Training may naturally regularize topology complexity")
        print(f"  - Different normalization behavior between trained/random weights")


if __name__ == "__main__":
    main()