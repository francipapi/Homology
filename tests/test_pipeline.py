import pytest
import numpy as np
import torch
from pathlib import Path
import time
import yaml # Added for loading config in new test

from src.data.dataset import generate
from src.models.trainer import train
from src.topology.homology import compute_persistent_homology
from src.utils.distance import compute_distance_matrix

# Test configuration
TEST_CONFIG = {
    'data': {
        'n': 20,  # Further reduced for faster testing
        'big_radius': 3,
        'small_radius': 1
    },
    'model': {
        'width': 4,  # Reduced for faster testing
        'layers': 2, # Corresponds to number of ReLU outputs
        'epochs': 2, # Minimal epochs for speed
        'batch_size': 8,
        'lr': 0.001,
        'device': "cpu",  # Force CPU for testing consistency
        'seed': 42        # Added seed for reproducibility tests
    },
    'homology': {
        'max_dimension': 1,
        'max_edge_length': 1.0
    }
}

def print_timing(start_time, operation_name):
    """Helper function to print timing information."""
    elapsed = time.time() - start_time
    print(f"⏱️  {operation_name} took {elapsed:.2f} seconds")

@pytest.fixture(scope="session")
def test_data():
    """Fixture to generate test data once for all tests."""
    print("\nGenerating test data...")
    start_time = time.time()
    try:
        X, y = generate(
            n=TEST_CONFIG['data']['n'],
            big_radius=TEST_CONFIG['data']['big_radius'],
            small_radius=TEST_CONFIG['data']['small_radius']
        )
        print_timing(start_time, "Data generation")
        return X, y
    except Exception as e:
        pytest.fail(f"Failed to generate test data: {str(e)}")

def test_data_generation(test_data):
    """Test dataset generation."""
    X, y = test_data
    print("\nTesting data generation...")
    start_time = time.time()
    
    try:
        assert X.shape[0] == 2 * TEST_CONFIG['data']['n']  # Points per torus
        assert X.shape[1] == 3    # 3D points
        assert y.shape[0] == 2 * TEST_CONFIG['data']['n']  # One label per point
        assert np.all(np.isin(y, [0, 1]))  # Binary labels
        print_timing(start_time, "Data validation")
        print("✓ Data generation test passed")
    except AssertionError as e:
        pytest.fail(f"Data generation test failed: {str(e)}")

def test_model_training_reproducibility(test_data):
    """Test neural network training for reproducibility with a fixed seed."""
    X, y = test_data
    print("\nTesting model training reproducibility...")
    
    # Use a copy of model params from TEST_CONFIG to ensure seed is included
    model_params = TEST_CONFIG['model'].copy()

    try:
        # First run
        print("  Starting first training run...")
        start_time_1 = time.time()
        layer_outputs_1 = train(
            X, y,
            **model_params # This now includes the seed
        )
        print_timing(start_time_1, "First model training run")
        
        assert len(layer_outputs_1) == model_params['layers']
        assert all(isinstance(output, np.ndarray) for output in layer_outputs_1)

        # Second run with the same seed
        print("  Starting second training run (same seed)...")
        start_time_2 = time.time()
        layer_outputs_2 = train(
            X, y,
            **model_params # This includes the same seed
        )
        print_timing(start_time_2, "Second model training run (same seed)")

        assert len(layer_outputs_2) == model_params['layers']
        # Check that all layer outputs are identical
        for i in range(len(layer_outputs_1)):
            assert np.array_equal(layer_outputs_1[i], layer_outputs_2[i]), \
                f"Layer {i} outputs are not identical between runs with the same seed."
        
        print("✓ Model training outputs are identical with the same seed.")

        # Optional: Test with a different seed to ensure outputs are different
        print("  Starting third training run (different seed)...")
        model_params_diff_seed = model_params.copy()
        model_params_diff_seed['seed'] = model_params['seed'] + 1 # Change the seed
        
        start_time_3 = time.time()
        layer_outputs_3 = train(X, y, **model_params_diff_seed)
        print_timing(start_time_3, "Third model training run (different seed)")
        
        assert len(layer_outputs_3) == model_params['layers']
        # Check if *any* layer output is different. For very small networks/data/epochs,
        # they might coincidentally be the same, but typically they should differ.
        are_different = False
        if len(layer_outputs_1) == len(layer_outputs_3):
            for i in range(len(layer_outputs_1)):
                if not np.array_equal(layer_outputs_1[i], layer_outputs_3[i]):
                    are_different = True
                    break
        else: # Should not happen if only seed changed
            are_different = True 

        assert are_different, "Model outputs were identical even with different seeds. This might be okay for very small models/data but is worth noting."
        print("✓ Model outputs differ with different seeds (as expected for this test setup).")
        print("✓ Model training reproducibility test passed")

    except Exception as e:
        pytest.fail(f"Model training reproducibility test failed: {str(e)}")

def test_homology_computation_file_output(tmp_path): # Added tmp_path fixture
    """Test persistent homology computation and file output."""
    print("\nTesting homology computation and file output...")
    
    # Define file paths for outputs within the temporary directory
    diag_file = tmp_path / "persistence_diagram.txt"
    barcode_file = tmp_path / "persistence_barcode.png"
    betti_file = tmp_path / "betti_numbers.txt"
    
    try:
        # Create a simple point cloud (e.g., a square, using 3D for consistency with dataset)
        points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]) 
        start_time = time.time()
        print("  Computing distance matrix for homology test...")
        dist_matrix = compute_distance_matrix(points)
        print_timing(start_time, "Distance matrix computation (homology test)")
        
        start_time = time.time()
        print("  Computing persistent homology with file outputs...")
        betti_numbers = compute_persistent_homology(
            dist_matrix,
            max_dimension=TEST_CONFIG['homology']['max_dimension'],
            max_edge_length=TEST_CONFIG['homology']['max_edge_length'],
            diag_filename=str(diag_file), # Ensure paths are passed as strings
            plot_barcode_filename=str(barcode_file),
            betti_filename=str(betti_file)
        )
        print_timing(start_time, "Homology computation with file outputs")
        
        assert len(betti_numbers) > 0, "Betti numbers list should not be empty."
        assert diag_file.exists(), "Persistence diagram file was not created."
        assert barcode_file.exists(), "Barcode plot file was not created."
        assert betti_file.exists(), "Betti numbers file was not created."
        
        print(f"✓ Homology output files created successfully in {tmp_path}")
        print("✓ Homology computation and file output test passed")
    except Exception as e:
        pytest.fail(f"Homology computation and file output test failed: {str(e)}")

def test_pipeline_integration(test_data):
    """Test the full pipeline integration."""
    X, y = test_data
    print("\nTesting full pipeline integration...")
    
    try:
        # Train model
        start_time = time.time()
        print("  Starting model training...")
        layer_outputs = train(
            X, y,
            **TEST_CONFIG['model']
        )
        print_timing(start_time, "Model training in pipeline")
        
        # Compute homology for first layer
        start_time = time.time()
        print("  Computing distance matrix...")
        dist_matrix = compute_distance_matrix(layer_outputs[0])
        print_timing(start_time, "Distance matrix computation")
        
        start_time = time.time()
        print("  Computing persistent homology...")
        betti = compute_persistent_homology(
            dist_matrix,
            max_dimension=TEST_CONFIG['homology']['max_dimension'],
            max_edge_length=TEST_CONFIG['homology']['max_edge_length']
        )
        print_timing(start_time, "Homology computation")
        
        assert len(betti) > 0  # Should have at least one Betti number
        print("✓ Pipeline integration test passed")
    except Exception as e:
        pytest.fail(f"Pipeline integration test failed: {str(e)}")

def test_main_config_integration():
    """
    Tests integration of parameters from training_config.yaml with generate() and train().
    This simulates parts of main.py's configuration loading and usage.
    """
    print("\nTesting main.py config file integration (data generation part)...")
    # Assuming tests are run from the project root where 'configs' directory is accessible.
    config_file_path = Path("configs/training_config.yaml")

    if not config_file_path.exists():
        pytest.skip(f"Configuration file {config_file_path} not found. Skipping config integration test. Ensure tests are run from project root.")
        return

    try:
        with open(config_file_path, 'r') as f:
            main_training_config = yaml.safe_load(f)
        
        # Verify essential keys for data generation are present from training_config.yaml
        assert 'data' in main_training_config, "Config Error: 'data' key missing"
        assert 'generation' in main_training_config['data'], "Config Error: 'data.generation' key missing"
        data_gen_params_from_config = main_training_config['data']['generation']
        assert 'n' in data_gen_params_from_config, "Config Error: 'data.generation.n' missing"
        assert 'big_radius' in data_gen_params_from_config, "Config Error: 'data.generation.big_radius' missing"
        assert 'small_radius' in data_gen_params_from_config, "Config Error: 'data.generation.small_radius' missing"

        # For testing purposes, override 'n' to keep data generation fast, but use other params from config
        test_n_value = TEST_CONFIG['data']['n'] 
        current_data_gen_params = {
            'n': test_n_value, # Use smaller n from TEST_CONFIG for speed
            'big_radius': data_gen_params_from_config['big_radius'],
            'small_radius': data_gen_params_from_config['small_radius']
        }

        print(f"  Config Integration: Generating data with n={test_n_value} using radii from main config...")
        start_time = time.time()
        X, y = generate(**current_data_gen_params)
        print_timing(start_time, "Config Integration: Data generation")
        
        # generate() creates 2*n points
        assert X.shape[0] == 2 * test_n_value, f"Config Integration: Generated X has {X.shape[0]} samples, expected {2*test_n_value}"
        assert y.shape[0] == 2 * test_n_value, f"Config Integration: Generated y has {y.shape[0]} samples, expected {2*test_n_value}"
        print("✓ Config Integration: Data generation part passed.")

        # Part 2: Model training using parameters from the loaded config
        print("  Config Integration: Starting model training using parameters from main config...")
        
        # Ensure model and training sections and seed are in the loaded config
        assert 'model' in main_training_config, "Config Error: 'model' key missing"
        assert 'training' in main_training_config, "Config Error: 'training' key missing"
        assert 'seed' in main_training_config['training'], "Config Error: 'training.seed' missing"
        assert 'device' in main_training_config['training'], "Config Error: 'training.device' missing"
        assert 'learning_rate' in main_training_config['training'], "Config Error: 'training.learning_rate' missing"

        # Use structure/epochs from TEST_CONFIG for speed, but seed/lr/device from main_training_config
        train_call_params = {
            'width': TEST_CONFIG['model']['width'], 
            'layers': TEST_CONFIG['model']['layers'],
            'epochs': TEST_CONFIG['model']['epochs'],
            'batch_size': TEST_CONFIG['model']['batch_size'], # Could also take from main_config if desired
            'lr': main_training_config['training']['learning_rate'],
            'device': main_training_config['training']['device'],
            'seed': main_training_config['training']['seed'] # Crucially use seed from main config
        }
        
        start_time = time.time()
        layer_outputs = train(X, y, **train_call_params)
        print_timing(start_time, "Config Integration: Model training")
        
        assert len(layer_outputs) == train_call_params['layers'], \
            f"Config Integration: Trained model produced {len(layer_outputs)} layer outputs, expected {train_call_params['layers']}"
        assert all(isinstance(output, np.ndarray) for output in layer_outputs), \
            "Config Integration: Layer outputs are not all numpy arrays"
        
        print("✓ Config Integration: Model training part passed.")
        print("✓ main.py config integration test passed successfully.")

    except FileNotFoundError: # Should be caught by the exists() check, but as a fallback.
        pytest.skip(f"Main configuration file {config_file_path} not found during test execution.")
    except Exception as e:
        pytest.fail(f"main.py config integration test failed: {str(e)}")


if __name__ == "__main__":
    # This allows running pytest on this file directly for debugging.
    # Adds -v for verbose output and -s to show print statements (captures stdout).
    pytest.main([__file__, "-v", "-s"])