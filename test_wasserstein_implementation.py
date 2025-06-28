#!/usr/bin/env python3
"""
Test script to verify the Wasserstein generalization implementation.
This script will:
1. Modify the training config to enable train_test_layer_extraction
2. Run a simple training with torch_mlp.py
3. Run the Wasserstein analysis on the extracted activations
"""

import subprocess
import sys
import os
import yaml
from pathlib import Path

def main():
    print("=" * 60)
    print("TESTING WASSERSTEIN GENERALIZATION IMPLEMENTATION")
    print("=" * 60)
    
    # Step 1: Create a test configuration with train_test_layer_extraction enabled
    print("\n1. Creating test configuration...")
    
    # Load the existing training config
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify settings for quick testing
    config['training']['epochs'] = 5  # Reduce epochs for quick test
    config['data']['generation']['n'] = 500  # Reduce data size
    config['layer_extraction']['enabled'] = True
    config['layer_extraction']['train_test_layer_extraction'] = True  # Enable our new feature
    
    # Save test config
    test_config_path = 'configs/test_wasserstein_config.yaml'
    with open(test_config_path, 'w') as f:
        yaml.safe_dump(config, f)
    
    print(f"Test configuration saved to: {test_config_path}")
    
    # Step 2: Run training with torch_mlp.py
    print("\n2. Running training with train/test layer extraction...")
    
    # Use the myenv Python interpreter as specified in CLAUDE.md
    python_path = '/opt/anaconda3/envs/myenv/bin/python'
    
    try:
        result = subprocess.run(
            [python_path, 'src/models/torch_mlp.py', test_config_path],
            capture_output=True,
            text=True,
            check=True
        )
        print("Training completed successfully!")
        print("Output:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print("Error output:", e.stderr)
        return 1
    
    # Step 3: Check if train/test layer outputs were created
    print("\n3. Checking for train/test layer outputs...")
    
    train_test_dir = Path('results/train_test_layer_outputs')
    if train_test_dir.exists():
        files = list(train_test_dir.glob('*.pt'))
        print(f"Found {len(files)} layer output files:")
        for f in files:
            print(f"  - {f.name}")
    else:
        print("ERROR: Train/test layer outputs directory not found!")
        return 1
    
    # Step 4: Run Wasserstein analysis
    print("\n4. Running Wasserstein generalization analysis...")
    
    try:
        result = subprocess.run(
            [python_path, 'src/analysis/wasserstein_generalization.py'],
            capture_output=True,
            text=True,
            check=True
        )
        print("Wasserstein analysis completed successfully!")
        print("Output:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Wasserstein analysis failed with error: {e}")
        print("Error output:", e.stderr)
        return 1
    
    # Step 5: Check results
    print("\n5. Checking Wasserstein analysis results...")
    
    results_dir = Path('results/wasserstein_analysis')
    if results_dir.exists():
        files = list(results_dir.glob('*'))
        print(f"Found {len(files)} result files:")
        for f in files:
            print(f"  - {f.name}")
    else:
        print("ERROR: Wasserstein results directory not found!")
        return 1
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Clean up test config
    if os.path.exists(test_config_path):
        os.remove(test_config_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())