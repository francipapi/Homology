#!/usr/bin/env python3
"""
Simple test for boundary training to debug issues
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add project root to path
sys.path.append('.')

# Import the trainer
from src.models.decision_boundary_trainer import DecisionBoundaryTrainer

def create_minimal_test_config():
    """Create minimal configs for testing."""
    
    # Minimal training config
    training_config = {
        'model': {
            'input_dim': 3,
            'num_hidden_layers': 2,  # Small for quick testing
            'hidden_dim': 8,         # Very small
            'output_dim': 1,
            'activation_fn_name': 'relu',
            'dropout_rate': 0.0,
            'use_batch_norm': False
        },
        'training': {
            'device': 'cpu',
            'epochs': 3,             # Very few epochs
            'batch_size': 32,
            'learning_rate': 0.01,
            'seed': 42,
            'optimizer': {'name': 'adam'},
            'regularization': {'l1_lambda': 0, 'l2_lambda': 0},
            'lr_scheduler': {'type': 'none'},
            'gradient_clipping': {'enabled': False},
            'early_stopping': {'enabled': False}
        },
        'data': {
            'type': 'synthetic',
            'generation': {
                'n': 500,            # Small dataset
                'big_radius': 3,
                'small_radius': 1,
                'solid': True,
                'interior_noise': 0.1
            },
            'split_ratio': 0.8
        },
        'layer_extraction': {
            'enabled': False         # Disable layer extraction
        }
    }
    
    # Minimal boundary config
    boundary_config = {
        'training': {
            'extraction_schedule': {
                'enabled': True,
                'frequency': 2,      # Extract at epochs 0 and 2
                'start_epoch': 0,
                'final_extraction': True
            }
        },
        'extraction': {
            'grid': {
                'resolution': [16, 16, 16],  # Very small grid
                'custom_bounds': {
                    'x_min': -4.0, 'x_max': 4.0,
                    'y_min': -4.0, 'y_max': 4.0,
                    'z_min': -4.0, 'z_max': 4.0
                }
            },
            'boundary_detection': {
                'threshold': 0.5,
                'tolerance': 0.2,    # Large tolerance
                'min_points': 10     # Very low minimum
            },
            'isosurface': {
                'enabled': False     # Disable isosurface to avoid errors
            },
            'point_sampling': {
                'enabled': True,
                'method': 'uniform', # Simple method
                'num_points': 200    # Few points
            }
        },
        'topology': {
            'computation': {
                'enabled': False     # Disabled - compute separately
            }
        },
        'output': {
            'directories': {
                'base_dir': 'test_boundary_output',
                'boundaries_dir': 'boundaries',
                'topology_dir': 'topology'
            },
            'storage': {
                'save_boundary_meshes': False,  # Disable mesh saving
                'save_topology_data': True
            }
        }
    }
    
    return training_config, boundary_config

def test_boundary_training():
    """Test boundary training with minimal config."""
    print("🧪 Testing Boundary Training")
    print("=" * 30)
    
    try:
        # Create test configs
        training_config, boundary_config = create_minimal_test_config()
        
        print("✅ Created test configurations")
        print(f"Model: {training_config['model']['num_hidden_layers']} layers, {training_config['model']['hidden_dim']} neurons")
        print(f"Training: {training_config['training']['epochs']} epochs")
        print(f"Data: {training_config['data']['generation']['n']} points")
        print(f"Grid: {boundary_config['extraction']['grid']['resolution']}")
        
        # Create trainer
        trainer = DecisionBoundaryTrainer(training_config, boundary_config)
        print("✅ Created trainer")
        
        # Run training
        print("\n🔥 Starting training...")
        results = trainer.train()
        
        print("✅ Training completed!")
        print(f"Final accuracy: {results['training_history']['test_accuracy'][-1]:.4f}")
        print(f"Boundary extractions: {len(results['boundary_results'])}")
        
        # Check boundary results
        for i, result in enumerate(results['boundary_results']):
            print(f"  Epoch {result.epoch}: {'✅' if result.success else '❌'} "
                  f"({len(result.boundary_points) if result.boundary_points is not None else 0} points)")
        
        print("\n🎯 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_boundary_training()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Test failed!")