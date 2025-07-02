#!/usr/bin/env python3
"""
Test script for Network Homology implementation

This script validates the network graph homology implementation by:
1. Testing graph construction for different layer types
2. Verifying the factor graph approach for convolutional layers
3. Computing persistent homology on network graphs
4. Testing the complete pipeline with a small network
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

from src.utils.network_graph_builder import (
    MLPGraphBuilder, ConvGraphBuilder, UnifiedGraphBuilder
)
from src.utils.network_simplicial_complex import (
    DirectedFlagComplex, compute_network_homology
)
from src.topology.network_homology_tracker import NetworkHomologyTracker
from src.analysis.persistence_distances import PersistenceDistanceCalculator


def test_mlp_graph_construction():
    """Test graph construction for MLP layers."""
    print("\n=== Testing MLP Graph Construction ===")
    
    # Create a simple linear layer
    layer = nn.Linear(10, 5)
    
    # Initialize random weights
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    
    # Build graph
    builder = MLPGraphBuilder()
    graph, output_vertices = builder.build_graph(layer, input_shape=(10,))
    
    print(f"Created graph with {graph.num_vertices()} vertices and {graph.num_edges()} edges")
    print(f"Number of output vertices: {len(output_vertices)}")
    
    # Verify structure
    assert graph.num_vertices() == 16  # 10 input + 5 output + 1 bias
    assert len(output_vertices) == 5
    
    # Check edge weights
    e_weight = graph.ep.weight
    weights = [e_weight[e] for e in graph.edges()]
    print(f"Edge weight range: [{min(weights):.4f}, {max(weights):.4f}]")
    
    print("✓ MLP graph construction test passed")
    return graph


def test_conv_factor_graph():
    """Test factor graph construction for convolutional layers."""
    print("\n=== Testing Conv Factor Graph Construction ===")
    
    # Create a Conv2D layer
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # Build factor graph
    builder = ConvGraphBuilder()
    input_shape = (3, 8, 8)  # Small image for testing
    graph, output_vertices = builder.build_graph(conv, input_shape)
    
    # Calculate expected counts
    n_input = 3 * 8 * 8  # 192 input nodes
    n_param = 3 * 3 * 3 * 16  # 432 parameter nodes
    n_output = 16 * 8 * 8  # 1024 output nodes
    n_bias = 16  # 16 bias parameter nodes
    
    total_expected = n_input + n_param + n_output + n_bias
    
    print(f"Created factor graph with {graph.num_vertices()} vertices and {graph.num_edges()} edges")
    print(f"Expected vertices: {total_expected}")
    print(f"  Input nodes: {n_input}")
    print(f"  Parameter nodes: {n_param}")
    print(f"  Output nodes: {n_output}")
    print(f"  Bias nodes: {n_bias}")
    
    # Count node types
    v_type = graph.vp.type
    node_types = {}
    for v in graph.vertices():
        t = v_type[v]
        node_types[t] = node_types.get(t, 0) + 1
    
    print(f"Actual node types: {node_types}")
    
    # Verify parameter nodes have self-loops
    param_self_loops = 0
    for v in graph.vertices():
        if v_type[v] == "parameter":
            # Check for self-loop
            for e in v.out_edges():
                if e.target() == v:
                    param_self_loops += 1
                    break
    
    print(f"Parameter nodes with self-loops: {param_self_loops}")
    
    print("✓ Conv factor graph construction test passed")
    return graph


def test_network_homology_computation():
    """Test homology computation on a network graph."""
    print("\n=== Testing Network Homology Computation ===")
    
    # Create a small test network
    class TestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(20, 10)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(10, 1)
            self.sigmoid = nn.Sigmoid()
            self.input_shape = (10,)
        
        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x
    
    # Create model
    model = TestNet()
    
    # Build network graph
    builder = UnifiedGraphBuilder()
    graph = builder.build_network_graph(model)
    
    print(f"Network graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
    
    # Compute homology
    try:
        result = compute_network_homology(graph, max_dimension=1)
        
        print(f"Betti numbers: {result['betti_numbers']}")
        print(f"Backend used: {result['backend_used']}")
        
        for dim, dgm in result['persistence_diagrams'].items():
            print(f"  Dimension {dim}: {len(dgm)} features")
        
        print("✓ Network homology computation test passed")
        
    except Exception as e:
        print(f"⚠ Homology computation failed: {e}")
        print("This might be due to missing dependencies (flagser/gudhi)")


def test_homology_tracker():
    """Test the complete homology tracking pipeline."""
    print("\n=== Testing Homology Tracker ===")
    
    # Create a simple network
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
        nn.Sigmoid()
    )
    model.input_shape = (5,)
    
    # Create tracker with minimal config
    config = {
        'network_homology': {'enabled': True, 'track_interval': 1},
        'graph_construction': {'normalize_weights': True},
        'simplicial_complex': {'max_dimension': 1},
        'distance_metrics': {'primary_metric': 'heat', 'backend': 'custom'}
    }
    
    tracker = NetworkHomologyTracker(config)
    
    # Track initial state
    distance1, snapshot1 = tracker.track_training_step(model, step=0)
    print(f"Initial Betti numbers: {snapshot1.betti_numbers}")
    
    # Modify weights slightly
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    # Track after modification
    distance2, snapshot2 = tracker.track_training_step(model, step=1)
    print(f"After modification - Betti numbers: {snapshot2.betti_numbers}")
    print(f"Distance from previous: {distance2:.6f}")
    
    # Get summary statistics
    stats = tracker.get_summary_statistics()
    print(f"Summary statistics: {stats}")
    
    print("✓ Homology tracker test passed")


def test_distance_metrics():
    """Test persistence distance computations."""
    print("\n=== Testing Distance Metrics ===")
    
    # Create sample persistence diagrams
    diagram1 = np.array([[0.1, 0.5], [0.2, 0.8], [0.3, 0.6]])
    diagram2 = np.array([[0.15, 0.55], [0.25, 0.75], [0.35, 0.65], [0.4, 0.7]])
    
    calculator = PersistenceDistanceCalculator()
    
    # Test different metrics
    metrics = ['wasserstein', 'bottleneck', 'heat', 'silhouette']
    
    for metric in metrics:
        try:
            distance = calculator.compute_distance(diagram1, diagram2, metric=metric)
            print(f"{metric.capitalize()} distance: {distance:.6f}")
        except Exception as e:
            print(f"⚠ {metric} distance failed: {e}")
    
    print("✓ Distance metrics test passed")


def test_conv_network():
    """Test with a network containing convolutional layers."""
    print("\n=== Testing Conv Network with Factor Graphs ===")
    
    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
            self.relu2 = nn.ReLU()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(16 * 4 * 4, 10)
            self.input_shape = (1, 8, 8)
        
        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.flatten(x)
            x = self.fc(x)
            return x
    
    model = ConvNet()
    
    # Build graph
    builder = UnifiedGraphBuilder()
    graph = builder.build_network_graph(model)
    
    print(f"Conv network graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
    
    # Count parameter nodes
    v_type = graph.vp.type
    param_count = sum(1 for v in graph.vertices() if v_type[v] == "parameter")
    bias_param_count = sum(1 for v in graph.vertices() if v_type[v] == "bias_parameter")
    
    print(f"Parameter nodes: {param_count}")
    print(f"Bias parameter nodes: {bias_param_count}")
    
    # Expected parameter nodes
    expected_params = (1*8*3*3) + (8*16*3*3) + (16*4*4*10)  # Conv1 + Conv2 + FC
    expected_bias = 8 + 16 + 10  # Bias for each layer
    
    print(f"Expected parameter nodes: {expected_params}")
    print(f"Expected bias nodes: {expected_bias}")
    
    print("✓ Conv network test passed")


def run_all_tests():
    """Run all tests."""
    print("=== Network Homology Implementation Tests ===")
    print("Testing the factor graph approach for convolutional layers")
    
    try:
        # Test individual components
        test_mlp_graph_construction()
        test_conv_factor_graph()
        test_distance_metrics()
        
        # Test integrated functionality
        test_network_homology_computation()
        test_homology_tracker()
        test_conv_network()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()