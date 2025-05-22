import pytest
import numpy as np
import torch
from pathlib import Path
import time
import yaml
import atexit
import signal
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys

from src.data.dataset import generate
from src.models.trainer import train
from src.topology.homology import compute_persistent_homology
from src.utils.distance import compute_distance_matrix

# Test configuration
TEST_CONFIG = {
    'data': {
        'n': 10,  # Reduced from 20 to 10 for faster testing
        'big_radius': 3,
        'small_radius': 1
    },
    'model': {
        'width': 2,  # Reduced from 4 to 2 for faster testing
        'layers': 1, # Reduced from 2 to 1 for faster testing
        'epochs': 1, # Reduced from 2 to 1 for faster testing
        'batch_size': 4, # Reduced from 8 to 4 for faster testing
        'lr': 0.001,
        'device': "cpu",  # Force CPU for testing consistency
        'seed': 42        # Added seed for reproducibility tests
    },
    'homology': {
        'max_dimension': 1,  # Keep at 1 for faster computation
        'max_edge_length': 0.5  # Reduced from 1.0 to 0.5 for faster computation
    }
}

# Global pool for multiprocessing
_pool = None

def cleanup_pool():
    """Cleanup function for multiprocessing pool."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool.join()
        _pool = None

def get_pool():
    """Get or create a multiprocessing pool."""
    global _pool
    if _pool is None:
        _pool = Pool(processes=min(cpu_count(), 4))  # Limit to 4 processes for testing
        atexit.register(cleanup_pool)
    return _pool

# Register cleanup handlers
atexit.register(cleanup_pool)

def print_timing(start_time, operation_name):
    """Helper function to print timing information."""
    elapsed = time.time() - start_time
    return f"⏱️  {operation_name} took {elapsed:.2f} seconds"

def print_test_header(test_name):
    """Print a formatted test header."""
    print("\n" + "="*80)
    print(f"🧪 Running Test: {test_name}")
    print("="*80)

def print_test_result(test_name, success, timing_info=None):
    """Print a formatted test result."""
    status = "✅ PASSED" if success else "❌ FAILED"
    print("\n" + "-"*80)
    print(f"Test: {test_name}")
    print(f"Status: {status}")
    if timing_info:
        print(timing_info)
    print("-"*80 + "\n")

@pytest.fixture(scope="session", autouse=True)
def setup_teardown():
    """Setup and teardown for all tests."""
    print("\n🚀 Starting Test Suite")
    print("="*80)
    # Setup
    yield
    # Teardown
    cleanup_pool()
    print("\n🏁 Test Suite Completed")
    print("="*80)

@pytest.fixture(scope="session")
def test_data():
    """Fixture to generate test data once for all tests."""
    print_test_header("Data Generation")
    start_time = time.time()
    try:
        with tqdm(total=100, desc="Generating test data", ncols=80) as pbar:
            X, y = generate(
                n=TEST_CONFIG['data']['n'],
                big_radius=TEST_CONFIG['data']['big_radius'],
                small_radius=TEST_CONFIG['data']['small_radius']
            )
            pbar.update(100)
        timing_info = print_timing(start_time, "Data generation")
        print(f"\nGenerated {X.shape[0]} points in 3D space")
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(timing_info)
        return X, y
    except Exception as e:
        pytest.fail(f"Failed to generate test data: {str(e)}")

def test_data_generation(test_data):
    """Test dataset generation."""
    print_test_header("Data Validation")
    X, y = test_data
    start_time = time.time()
    
    try:
        with tqdm(total=4, desc="Validating data", ncols=80) as pbar:
            assert X.shape[0] == 2 * TEST_CONFIG['data']['n']  # Points per torus
            pbar.update(1)
            assert X.shape[1] == 3    # 3D points
            pbar.update(1)
            assert y.shape[0] == 2 * TEST_CONFIG['data']['n']  # One label per point
            pbar.update(1)
            assert np.all(np.isin(y, [0, 1]))  # Binary labels
            pbar.update(1)
        
        timing_info = print_timing(start_time, "Data validation")
        print_test_result("Data Generation", True, timing_info)
    except AssertionError as e:
        print_test_result("Data Generation", False)
        pytest.fail(f"Data generation test failed: {str(e)}")

def test_model_training_reproducibility(test_data):
    """Test neural network training for reproducibility with a fixed seed."""
    print_test_header("Model Training Reproducibility")
    X, y = test_data
    model_params = TEST_CONFIG['model'].copy()

    try:
        # First run
        print("\nTraining Run 1 (Original Seed)")
        start_time_1 = time.time()
        with tqdm(total=model_params['epochs'], desc="Training Run 1", ncols=80) as pbar:
            layer_outputs_1 = train(X, y, **model_params)
            pbar.update(model_params['epochs'])
        timing_1 = print_timing(start_time_1, "First model training run")
        print(timing_1)

        # Second run with same seed
        print("\nTraining Run 2 (Same Seed)")
        start_time_2 = time.time()
        with tqdm(total=model_params['epochs'], desc="Training Run 2", ncols=80) as pbar:
            layer_outputs_2 = train(X, y, **model_params)
            pbar.update(model_params['epochs'])
        timing_2 = print_timing(start_time_2, "Second model training run")
        print(timing_2)

        # Verify outputs
        print("\nVerifying outputs...")
        with tqdm(total=len(layer_outputs_1), desc="Verifying outputs", ncols=80) as pbar:
            for i in range(len(layer_outputs_1)):
                assert np.array_equal(layer_outputs_1[i], layer_outputs_2[i]), \
                    f"Layer {i} outputs are not identical between runs with the same seed."
                pbar.update(1)

        # Third run with different seed
        print("\nTraining Run 3 (Different Seed)")
        model_params_diff_seed = model_params.copy()
        model_params_diff_seed['seed'] = model_params['seed'] + 1
        start_time_3 = time.time()
        with tqdm(total=model_params['epochs'], desc="Training Run 3", ncols=80) as pbar:
            layer_outputs_3 = train(X, y, **model_params_diff_seed)
            pbar.update(model_params['epochs'])
        timing_3 = print_timing(start_time_3, "Third model training run")
        print(timing_3)

        # Verify outputs are different
        print("\nVerifying outputs are different...")
        are_different = False
        with tqdm(total=len(layer_outputs_1), desc="Checking differences", ncols=80) as pbar:
            for i in range(len(layer_outputs_1)):
                if not np.array_equal(layer_outputs_1[i], layer_outputs_3[i]):
                    are_different = True
                pbar.update(1)

        assert are_different, "Model outputs were identical even with different seeds."
        
        print_test_result("Model Training Reproducibility", True, 
                         f"{timing_1}\n{timing_2}\n{timing_3}")

    except Exception as e:
        print_test_result("Model Training Reproducibility", False)
        pytest.fail(f"Model training reproducibility test failed: {str(e)}")

def test_homology_computation_file_output(tmp_path):
    """Test persistent homology computation and file output."""
    print_test_header("Homology Computation")
    
    # Define file paths for outputs within the temporary directory
    diag_file = tmp_path / "persistence_diagram.txt"
    barcode_file = tmp_path / "persistence_barcode.png"
    betti_file = tmp_path / "betti_numbers.txt"
    
    try:
        # Create a simple point cloud
        print("\nCreating test point cloud...")
        points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]) 
        
        # Compute distance matrix
        print("\nComputing distance matrix...")
        start_time = time.time()
        with tqdm(total=1, desc="Distance matrix", ncols=80) as pbar:
            dist_matrix = compute_distance_matrix(points)
            pbar.update(1)
        timing_1 = print_timing(start_time, "Distance matrix computation")
        print(timing_1)
        
        # Compute persistent homology
        print("\nComputing persistent homology...")
        start_time = time.time()
        with tqdm(total=1, desc="Homology computation", ncols=80) as pbar:
            betti_numbers = compute_persistent_homology(
                dist_matrix,
                max_dimension=TEST_CONFIG['homology']['max_dimension'],
                max_edge_length=TEST_CONFIG['homology']['max_edge_length'],
                diag_filename=str(diag_file),
                plot_barcode_filename=str(barcode_file),
                betti_filename=str(betti_file)
            )
            pbar.update(1)
        timing_2 = print_timing(start_time, "Homology computation")
        print(timing_2)
        
        # Verify outputs
        print("\nVerifying output files...")
        with tqdm(total=4, desc="File verification", ncols=80) as pbar:
            assert len(betti_numbers) > 0, "Betti numbers list should not be empty."
            pbar.update(1)
            assert diag_file.exists(), "Persistence diagram file was not created."
            pbar.update(1)
            assert barcode_file.exists(), "Barcode plot file was not created."
            pbar.update(1)
            assert betti_file.exists(), "Betti numbers file was not created."
            pbar.update(1)
        
        print(f"\nOutput files created in: {tmp_path}")
        print_test_result("Homology Computation", True, f"{timing_1}\n{timing_2}")
    except Exception as e:
        print_test_result("Homology Computation", False)
        pytest.fail(f"Homology computation test failed: {str(e)}")

def test_pipeline_integration(test_data):
    """Test the full pipeline integration."""
    print_test_header("Pipeline Integration")
    X, y = test_data
    
    try:
        # Train model
        print("\nTraining model...")
        start_time = time.time()
        with tqdm(total=TEST_CONFIG['model']['epochs'], desc="Model training", ncols=80) as pbar:
            layer_outputs = train(X, y, **TEST_CONFIG['model'])
            pbar.update(TEST_CONFIG['model']['epochs'])
        timing_1 = print_timing(start_time, "Model training")
        print(timing_1)
        
        # Compute distance matrix
        print("\nComputing distance matrix...")
        start_time = time.time()
        with tqdm(total=1, desc="Distance matrix", ncols=80) as pbar:
            dist_matrix = compute_distance_matrix(layer_outputs[0])
            pbar.update(1)
        timing_2 = print_timing(start_time, "Distance matrix computation")
        print(timing_2)
        
        # Compute homology
        print("\nComputing persistent homology...")
        start_time = time.time()
        with tqdm(total=1, desc="Homology computation", ncols=80) as pbar:
            betti = compute_persistent_homology(
                dist_matrix,
                max_dimension=TEST_CONFIG['homology']['max_dimension'],
                max_edge_length=TEST_CONFIG['homology']['max_edge_length']
            )
            pbar.update(1)
        timing_3 = print_timing(start_time, "Homology computation")
        print(timing_3)
        
        assert len(betti) > 0
        print_test_result("Pipeline Integration", True, 
                         f"{timing_1}\n{timing_2}\n{timing_3}")
    except Exception as e:
        print_test_result("Pipeline Integration", False)
        pytest.fail(f"Pipeline integration test failed: {str(e)}")

def test_main_config_integration():
    """Tests integration of parameters from training_config.yaml."""
    print_test_header("Config Integration")
    config_file_path = Path("configs/training_config.yaml")

    if not config_file_path.exists():
        print_test_result("Config Integration", False, "Config file not found")
        pytest.skip(f"Configuration file {config_file_path} not found. Skipping config integration test.")
        return

    try:
        # Load config
        print("\nLoading configuration...")
        with tqdm(total=1, desc="Loading config", ncols=80) as pbar:
            with open(config_file_path, 'r') as f:
                main_training_config = yaml.safe_load(f)
            pbar.update(1)
        
        # Verify config structure
        print("\nVerifying config structure...")
        with tqdm(total=5, desc="Config validation", ncols=80) as pbar:
            assert 'data' in main_training_config, "Config Error: 'data' key missing"
            pbar.update(1)
            assert 'generation' in main_training_config['data'], "Config Error: 'data.generation' key missing"
            pbar.update(1)
            data_gen_params_from_config = main_training_config['data']['generation']
            assert 'n' in data_gen_params_from_config, "Config Error: 'data.generation.n' missing"
            pbar.update(1)
            assert 'big_radius' in data_gen_params_from_config, "Config Error: 'data.generation.big_radius' missing"
            pbar.update(1)
            assert 'small_radius' in data_gen_params_from_config, "Config Error: 'data.generation.small_radius' missing"
            pbar.update(1)

        # Generate data
        print("\nGenerating test data...")
        test_n_value = TEST_CONFIG['data']['n']
        current_data_gen_params = {
            'n': test_n_value,
            'big_radius': data_gen_params_from_config['big_radius'],
            'small_radius': data_gen_params_from_config['small_radius']
        }
        
        start_time = time.time()
        with tqdm(total=100, desc="Data generation", ncols=80) as pbar:
            X, y = generate(**current_data_gen_params)
            pbar.update(100)
        timing_1 = print_timing(start_time, "Data generation")
        print(timing_1)
        
        # Verify data
        print("\nVerifying generated data...")
        with tqdm(total=2, desc="Data verification", ncols=80) as pbar:
            assert X.shape[0] == 2 * test_n_value
            pbar.update(1)
            assert y.shape[0] == 2 * test_n_value
            pbar.update(1)
        
        # Train model
        print("\nTraining model with config parameters...")
        train_call_params = {
            'width': TEST_CONFIG['model']['width'],
            'layers': TEST_CONFIG['model']['layers'],
            'epochs': TEST_CONFIG['model']['epochs'],
            'batch_size': TEST_CONFIG['model']['batch_size'],
            'lr': main_training_config['training']['learning_rate'],
            'device': main_training_config['training']['device'],
            'seed': main_training_config['training']['seed']
        }
        
        start_time = time.time()
        with tqdm(total=train_call_params['epochs'], desc="Model training", ncols=80) as pbar:
            layer_outputs = train(X, y, **train_call_params)
            pbar.update(train_call_params['epochs'])
        timing_2 = print_timing(start_time, "Model training")
        print(timing_2)
        
        # Verify outputs
        print("\nVerifying model outputs...")
        with tqdm(total=2, desc="Output verification", ncols=80) as pbar:
            assert len(layer_outputs) == train_call_params['layers']
            pbar.update(1)
            assert all(isinstance(output, np.ndarray) for output in layer_outputs)
            pbar.update(1)
        
        print_test_result("Config Integration", True, f"{timing_1}\n{timing_2}")

    except Exception as e:
        print_test_result("Config Integration", False)
        pytest.fail(f"Config integration test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])