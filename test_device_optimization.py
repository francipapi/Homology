#!/usr/bin/env python3
"""
Test script for device-specific hyperparameter optimization.
Tests both CPU multiprocessing and GPU vectorized optimization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.utils.hyperparameter_optimization import HyperparameterOptimizer

def test_cpu_optimization():
    """Test CPU multiprocessing optimization."""
    print("🧪 Testing CPU multiprocessing optimization...")
    
    opt_config = {
        'n_trials': 4,
        'device': 'cpu',
        'cpu': {'n_jobs': 2},
        'gpu': {'vectorized_batch_size': 4},
        'study_name': 'test_cpu_optimization',
        'storage': 'sqlite:///test_cpu_optimization.db'
    }
    
    try:
        optimizer = HyperparameterOptimizer('configs/training_config.yaml', opt_config)
        optimizer.optimize(n_trials=4)
        
        if len(optimizer.study.trials) >= 4:
            print(f"✅ CPU test passed! Completed {len(optimizer.study.trials)} trials")
            print(f"   Best accuracy: {optimizer.study.best_value:.4f}")
            return True
        else:
            print("❌ CPU test failed: Not enough trials completed")
            return False
            
    except Exception as e:
        print(f"❌ CPU test failed with error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists('test_cpu_optimization.db'):
            os.remove('test_cpu_optimization.db')

def test_gpu_optimization():
    """Test GPU vectorized optimization (if available)."""
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("⚠️  No GPU available, skipping GPU test")
        return True
    
    print("🧪 Testing GPU vectorized optimization...")
    
    opt_config = {
        'n_trials': 8,
        'device': 'auto',  # Will auto-detect GPU
        'cpu': {'n_jobs': 2},
        'gpu': {'vectorized_batch_size': 4},
        'study_name': 'test_gpu_optimization',
        'storage': 'sqlite:///test_gpu_optimization.db'
    }
    
    try:
        optimizer = HyperparameterOptimizer('configs/training_config.yaml', opt_config)
        
        # Should use GPU if available
        if optimizer.device.type not in ['cuda', 'mps']:
            print("⚠️  GPU not detected, testing sequential on CPU instead")
        
        optimizer.optimize(n_trials=8)
        
        if len(optimizer.study.trials) >= 8:
            print(f"✅ GPU test passed! Completed {len(optimizer.study.trials)} trials")
            print(f"   Strategy used: {optimizer.optimization_strategy}")
            print(f"   Device: {optimizer.device}")
            print(f"   Best accuracy: {optimizer.study.best_value:.4f}")
            return True
        else:
            print("❌ GPU test failed: Not enough trials completed")
            return False
            
    except Exception as e:
        print(f"❌ GPU test failed with error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists('test_gpu_optimization.db'):
            os.remove('test_gpu_optimization.db')

def main():
    """Run all optimization tests."""
    print("🚀 Testing Device-Specific Hyperparameter Optimization")
    print("="*60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test CPU optimization
    if test_cpu_optimization():
        tests_passed += 1
    
    print()
    
    # Test GPU optimization
    if test_gpu_optimization():
        tests_passed += 1
    
    print("\n" + "="*60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Device-specific optimization is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)