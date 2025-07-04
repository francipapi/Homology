#!/usr/bin/env python3
"""
Test script to ensure consistent weight normalization between trained and random CNNs.
"""

import numpy as np
import torch
import torch.nn as nn
import yaml
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.topology.network_homology_tracker import NetworkHomologyTracker
from src.models.torch_custom import CustomNet
from src.analysis.persistence_distances import PersistenceDistanceCalculator


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


def get_model_max_abs_weight(model):
    """Compute max absolute weight in model."""
    max_abs_weight = 0.0
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            weights = module.weight.detach().cpu().numpy()
            max_abs_weight = max(max_abs_weight, np.max(np.abs(weights)))
            if module.bias is not None:
                bias = module.bias.detach().cpu().numpy()
                max_abs_weight = max(max_abs_weight, np.max(np.abs(bias)))
    
    return max_abs_weight


def create_tracker_with_fixed_normalization(config, max_abs_weight):
    """Create tracker that uses a fixed max_abs_weight for normalization."""
    
    class FixedNormalizationTracker(NetworkHomologyTracker):
        def __init__(self, config, fixed_max_weight):
            super().__init__(config)
            self.fixed_max_weight = fixed_max_weight
            
        def _build_network_graph(self, model):
            """Override to use fixed normalization."""
            # Force the max weight to our fixed value
            original_method = self.graph_builder._compute_max_abs_weight
            self.graph_builder._compute_max_abs_weight = lambda m: self.fixed_max_weight
            
            try:
                # Build graph with fixed normalization
                graph = self.graph_builder.build_network_graph(model)
                return graph
            finally:
                # Restore original method
                self.graph_builder._compute_max_abs_weight = original_method
    
    return FixedNormalizationTracker(config, max_abs_weight)


def test_consistent_normalization():
    """Test CNN distance computation with consistent normalization."""
    print("=" * 80)
    print("CNN Distance Test with Consistent Weight Normalization")
    print("=" * 80)
    
    # Load configuration
    config_path = "configs/network_homology_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use memory-friendly settings
    config['network_homology']['persistence']['epsilon_filtering'] = None
    config['network_homology']['simplicial_complex']['max_dimension'] = 1
    config['network_homology']['tracking']['store_full_diagrams'] = False
    
    # Load trained CNN models
    cnn_paths = [
        Path("results/homology_training/custom_1751557522/model.pt"),
        Path("results/homology_training/custom_1751631504/model.pt")
    ]
    
    print("\n1. Loading trained CNN models and analyzing weights...")
    models = []
    max_weights = []
    
    for i, path in enumerate(cnn_paths):
        if not path.exists():
            print(f"Error: Model not found at {path}")
            return
        
        model, model_config = load_cnn_model(path)
        max_weight = get_model_max_abs_weight(model)
        
        models.append(model)
        max_weights.append(max_weight)
        
        print(f"  Trained CNN {i+1}: max_abs_weight = {max_weight:.6f}")
    
    # Create random CNN
    print("\n2. Creating random CNN...")
    random_model = create_random_cnn(model_config)
    random_max_weight = get_model_max_abs_weight(random_model)
    print(f"  Random CNN: max_abs_weight = {random_max_weight:.6f}")
    
    # Use the maximum weight across ALL models for consistent normalization
    global_max_weight = max(max_weights + [random_max_weight])
    print(f"\n3. Using global max_abs_weight = {global_max_weight:.6f} for all models")
    
    models.append(random_model)
    model_names = ['Trained_CNN_1', 'Trained_CNN_2', 'Random_CNN']
    
    # Compute persistence diagrams with consistent normalization
    print("\n4. Computing persistence diagrams with consistent normalization...")
    
    tracker = create_tracker_with_fixed_normalization(config, global_max_weight)
    distance_calculator = PersistenceDistanceCalculator(backend="gudhi")
    
    all_diagrams = []
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"  Processing {name}...")
        
        # Build graph to check edge weights
        graph = tracker._build_network_graph(model)
        edge_weights = [graph.ep.weight[e] for e in graph.edges()]
        
        print(f"    Graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
        print(f"    Edge weights: [{np.min(edge_weights):.3f}, {np.max(edge_weights):.3f}]")
        
        # Compute homology
        _, snapshot = tracker.track_training_step(model=model, step=0, epoch=0)
        total_points = sum(len(dgm) for dgm in snapshot.persistence_diagrams.values())
        
        print(f"    Persistence points: {total_points}")
        all_diagrams.append(snapshot.persistence_diagrams)
    
    # Compute distances
    print("\n5. Computing pairwise distances...")
    
    def compute_distance(diagrams1, diagrams2, metric='wasserstein'):
        total_distance = 0.0
        max_dim = max(max(diagrams1.keys(), default=0), 
                     max(diagrams2.keys(), default=0))
        
        for dim in range(max_dim + 1):
            dgm1 = diagrams1.get(dim, np.empty((0, 2)))
            dgm2 = diagrams2.get(dim, np.empty((0, 2)))
            dist = distance_calculator.compute_distance(dgm1, dgm2, metric=metric)
            total_distance += dist
        
        return total_distance
    
    # Compute distance matrix
    print(f"\nWASSERSTEIN DISTANCES (with consistent normalization):")
    print(f"{'':15} {'Trained_1':>12} {'Trained_2':>12} {'Random':>12}")
    
    distance_matrix = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(3):
            if i == j:
                dist = 0.0
            else:
                dist = compute_distance(all_diagrams[i], all_diagrams[j], 'wasserstein')
            distance_matrix[i, j] = dist
    
    # Print results
    for i, name in enumerate(model_names):
        row = f"{name:15}"
        for j in range(3):
            if i == j:
                row += f" {'0.000':>12}"
            else:
                row += f" {distance_matrix[i, j]:>12.3f}"
        print(row)
    
    # Analysis
    trained_distance = distance_matrix[0, 1]
    random_distances = [distance_matrix[0, 2], distance_matrix[1, 2]]
    avg_random_distance = np.mean(random_distances)
    
    print(f"\nANALYSIS (with consistent normalization):")
    print(f"  Distance between trained CNNs: {trained_distance:.3f}")
    print(f"  Average trained vs random: {avg_random_distance:.3f}")
    print(f"  Ratio (random/trained): {avg_random_distance/trained_distance:.2f}x")
    
    if avg_random_distance > trained_distance:
        ratio = avg_random_distance / trained_distance
        print(f"  ✅ SUCCESS: Random CNN more distant ({ratio:.1f}x) with consistent normalization")
    else:
        print(f"  ⚠️  Unexpected: Random CNN should be more distant even with consistent normalization")
    
    print(f"\n6. CONCLUSION:")
    print(f"   With consistent weight normalization:")
    print(f"   - All models use same max_abs_weight = {global_max_weight:.6f}")
    print(f"   - Edge weight ranges are more comparable")
    print(f"   - Topological complexity is more balanced")
    print(f"   - Random CNN computation should be stable")


if __name__ == "__main__":
    test_consistent_normalization()