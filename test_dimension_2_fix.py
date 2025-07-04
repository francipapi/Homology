#!/usr/bin/env python3
"""
Test dimension 2 computation with consistent normalization to fix RAM explosion.
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


def test_dimension_2_computation():
    """Test if dimension 2 computation works with consistent normalization."""
    print("=" * 80)
    print("Testing Dimension 2 Computation with Consistent Normalization")
    print("=" * 80)
    
    # Load configuration
    config_path = "configs/network_homology_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['network_homology']['persistence']['epsilon_filtering'] = None
    config['network_homology']['simplicial_complex']['max_dimension'] = 2
    
    # Load trained CNN
    model_path = Path("results/homology_training/custom_1751557522/model.pt")
    if not model_path.exists():
        print("Trained model not found!")
        return
    
    print("1. Loading trained CNN...")
    trained_model, model_config = load_cnn_model(model_path)
    trained_max_weight = get_model_max_abs_weight(trained_model)
    print(f"   Trained CNN max_abs_weight: {trained_max_weight:.6f}")
    
    print("2. Creating random CNN...")
    random_model = create_random_cnn(model_config)
    random_max_weight = get_model_max_abs_weight(random_model)
    print(f"   Random CNN max_abs_weight: {random_max_weight:.6f}")
    
    # Use global max weight for consistent normalization
    global_max_weight = max(trained_max_weight, random_max_weight)
    print(f"3. Using global max_abs_weight: {global_max_weight:.6f}")
    
    # Test trained CNN with dimension 2
    print("\n4. Testing trained CNN with dimension 2...")
    trained_tracker = create_tracker_with_fixed_normalization(config, global_max_weight)
    
    try:
        _, trained_snapshot = trained_tracker.track_training_step(model=trained_model, step=0, epoch=0)
        trained_total_points = sum(len(dgm) for dgm in trained_snapshot.persistence_diagrams.values())
        print(f"   ✅ SUCCESS: {trained_total_points} total persistence points")
        
        for dim, dgm in trained_snapshot.persistence_diagrams.items():
            print(f"     Dimension {dim}: {len(dgm)} points")
            
    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        return
    
    # Test random CNN with dimension 2
    print("\n5. Testing random CNN with dimension 2...")
    random_tracker = create_tracker_with_fixed_normalization(config, global_max_weight)
    
    try:
        _, random_snapshot = random_tracker.track_training_step(model=random_model, step=0, epoch=0)
        random_total_points = sum(len(dgm) for dgm in random_snapshot.persistence_diagrams.values())
        print(f"   ✅ SUCCESS: {random_total_points} total persistence points")
        
        for dim, dgm in random_snapshot.persistence_diagrams.items():
            print(f"     Dimension {dim}: {len(dgm)} points")
            
    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        print(f"   This indicates RAM explosion was NOT fixed by consistent normalization")
        return
    
    print("\n6. CONCLUSION:")
    print("   ✅ Both trained and random CNNs work with dimension 2!")
    print("   ✅ Consistent normalization successfully prevents RAM explosion")
    print("   ✅ Random CNN computation is now stable")


if __name__ == "__main__":
    test_dimension_2_computation()