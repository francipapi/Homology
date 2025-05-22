import pytest
import numpy as np
import torch
from pathlib import Path
import time
from tqdm import tqdm

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
        'layers': 2,
        'epochs': 2,
        'batch_size': 8,
        'lr': 0.001,
        'device': "cpu"  # Force CPU for testing
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

def test_model_training(test_data):
    """Test neural network training."""
    X, y = test_data
    print("\nTesting model training...")
    
    try:
        start_time = time.time()
        print("  Starting model training...")
        layer_outputs = train(
            X, y,
            **TEST_CONFIG['model']
        )
        print_timing(start_time, "Model training")
        
        assert len(layer_outputs) == TEST_CONFIG['model']['layers']
        assert all(isinstance(output, np.ndarray) for output in layer_outputs)
        print("✓ Model training test passed")
    except Exception as e:
        pytest.fail(f"Model training test failed: {str(e)}")

def test_homology_computation():
    """Test persistent homology computation."""
    print("\nTesting homology computation...")
    
    try:
        # Create a simple point cloud (square)
        points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        start_time = time.time()
        print("  Computing distance matrix...")
        dist_matrix = compute_distance_matrix(points)
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
        print("✓ Homology computation test passed")
    except Exception as e:
        pytest.fail(f"Homology computation test failed: {str(e)}")

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

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 