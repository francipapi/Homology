#!/usr/bin/env python3
"""
Quick test script for hyperparameter optimization.
Runs 3 trials to verify everything works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.hyperparameter_optimization import HyperparameterOptimizer

def test_optimization():
    """Run a quick test with 3 trials."""
    print("🧪 Testing hyperparameter optimization with 3 trials...")
    
    # Test configuration
    opt_config = {
        'n_trials': 3,
        'n_jobs': 1,  # Use single job for testing
        'study_name': 'test_optimization',
        'storage': 'sqlite:///test_optimization.db'
    }
    
    try:
        # Initialize optimizer
        optimizer = HyperparameterOptimizer('configs/training_config.yaml', opt_config)
        
        # Run optimization
        optimizer.optimize(n_trials=3, n_jobs=1)
        
        # Check results
        if len(optimizer.study.trials) >= 3:
            print(f"✅ Test passed! Completed {len(optimizer.study.trials)} trials")
            print(f"   Best accuracy: {optimizer.study.best_value:.4f}")
            print(f"   Best params: {optimizer.study.best_params}")
            
            # Save test config
            optimizer.save_best_config('configs/test_optimized_parameters.yaml')
            print("   Saved test config to: configs/test_optimized_parameters.yaml")
            
            return True
        else:
            print("❌ Test failed: Not enough trials completed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        # Clean up test database
        import os
        if os.path.exists('test_optimization.db'):
            os.remove('test_optimization.db')
            print("🧹 Cleaned up test database")

if __name__ == "__main__":
    success = test_optimization()
    sys.exit(0 if success else 1)