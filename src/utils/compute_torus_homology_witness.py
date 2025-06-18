"""
Compute homology of the torus training dataset using Witness Complexes.

This script generates the torus dataset using the same parameters as the training
pipeline and computes persistent homology using witness complexes with optimized settings
from homology_config.yaml. It provides a direct comparison to the Gudhi and Ripser
approaches for the same dataset.

Features:
- Uses witness complex construction instead of Vietoris-Rips
- Landmark selection with maxmin or FPS algorithms
- Configurable relaxation and filtration parameters
- Parallel processing support
- Progress tracking and performance monitoring
- Fallback strategies for robustness

Usage:
    python src/utils/compute_torus_homology_witness.py
    python src/utils/compute_torus_homology_witness.py --parallel
    python src/utils/compute_torus_homology_witness.py --config custom_config.yaml
"""

import numpy as np
import yaml
import time
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import argparse
import warnings
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.data.dataset import generate
    from src.topology.compute_homology_witness import process_single_layer_witness_optimized
    print("✅ Successfully imported witness complex computation functions")
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("   Make sure you're running this in the myenv conda environment")
    sys.exit(1)


def load_configs() -> Tuple[Dict, Dict]:
    """
    Load training and homology configuration files.
    
    Returns:
        Tuple of (training_config, homology_config) dictionaries
    """
    # Load training configuration
    training_config_path = Path("configs/training_config.yaml")
    if not training_config_path.exists():
        raise FileNotFoundError(f"Training config not found: {training_config_path}")
    
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    # Load homology configuration
    homology_config_path = Path("configs/homology_config.yaml")
    if not homology_config_path.exists():
        raise FileNotFoundError(f"Homology config not found: {homology_config_path}")
    
    with open(homology_config_path, 'r') as f:
        homology_config = yaml.safe_load(f)
    
    return training_config, homology_config


def generate_torus_dataset(training_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Generate torus dataset using training configuration parameters."""
    print("🌍 Generating torus dataset...")
    data_params = training_config['data']['generation']
    
    X, y = generate(
        n=data_params['n'],
        big_radius=data_params['big_radius'],
        small_radius=data_params['small_radius'],
        solid=data_params.get('solid', False),
        interior_noise=data_params.get('interior_noise', 0.1)
    )
    
    print(f"   ✅ Generated dataset: {X.shape[0]} points, {X.shape[1]} dimensions")
    print(f"   ✅ Classes: {np.unique(y.flatten())}")
    print(f"   ✅ Points per class: {np.bincount(y.astype(int).flatten())}")
    print(f"   ✅ Torus type: {'Solid' if data_params.get('solid', False) else 'Hollow'}")
    if data_params.get('solid', False):
        print(f"   ✅ Interior noise: {data_params.get('interior_noise', 0.1)}")
    
    return X, y


def prepare_dataset_for_witness(X: np.ndarray, homology_config: Dict) -> np.ndarray:
    """
    Prepare dataset for witness complex computation.
    Witness complexes handle their own sampling internally, so no external sampling needed.
    """
    print(f"   ✅ Using full dataset: {X.shape[0]} points")
    return X


def compute_witness_homology(X: np.ndarray, homology_config: Dict, use_parallel: bool = True) -> Dict:
    """
    Compute persistent homology using witness complexes.
    
    Args:
        X: Point cloud data (n_points, n_dimensions)
        homology_config: Configuration dictionary
        use_parallel: Whether to use parallel processing
    
    Returns:
        Dictionary containing homology results
    """
    print("\n🔬 WITNESS COMPLEX HOMOLOGY COMPUTATION")
    print("=" * 50)
    
    start_time = time.time()
    
    # Extract witness complex configuration
    witness_config = homology_config['witness_complex']
    computation_config = homology_config['computation']
    
    print(f"📊 Dataset statistics:")
    print(f"   • Points: {X.shape[0]}")
    print(f"   • Dimensions: {X.shape[1]}")
    print(f"   • Memory usage: ~{X.nbytes / 1024**2:.1f} MB")
    
    print(f"\n⚙️ Witness complex parameters:")
    
    # Calculate actual number of landmarks that will be used
    n_points = X.shape[0]
    if witness_config.get('adaptive_landmarks', False):
        landmark_percentage = witness_config.get('landmark_percentage', 0.005)
        calculated_landmarks = int(n_points * landmark_percentage)
        min_landmarks = witness_config.get('min_landmarks', 20)
        max_landmarks = witness_config.get('max_landmarks', 200)
        actual_landmarks = max(min_landmarks, min(calculated_landmarks, max_landmarks))
        
        print(f"   • Adaptive landmarks: ENABLED")
        print(f"   • Landmark percentage: {landmark_percentage} ({landmark_percentage*100:.1f}%)")
        print(f"   • Calculated landmarks: {calculated_landmarks} (from {n_points} points)")
        print(f"   • Min/Max limits: {min_landmarks}/{max_landmarks}")
        print(f"   • Final landmarks: {actual_landmarks}")
    else:
        actual_landmarks = witness_config.get('n_landmarks', 50)
        print(f"   • Adaptive landmarks: DISABLED")
        print(f"   • Fixed landmarks: {actual_landmarks}")
    
    print(f"   • Selection method: {witness_config['landmark_selection']}")
    if witness_config['landmark_selection'] == 'maxmin':
        init_strategy = witness_config.get('maxmin_init_strategy', 'center')
        print(f"   • Maxmin init strategy: {init_strategy}")
        add_noise = witness_config.get('maxmin_add_noise', False)
        if add_noise:
            noise_scale = witness_config.get('maxmin_noise_scale', 1e-6)
            print(f"   • Maxmin noise: {noise_scale}")
    
    print(f"   • Max witnesses: {witness_config['max_witnesses']}")
    print(f"   • Witness threshold: {witness_config.get('witness_threshold', 60000)}")
    
    # Check if witness sampling will be used
    witness_threshold = witness_config.get('witness_threshold', 60000)
    use_witness_sampling = witness_config.get('use_witness_sampling', True)
    if use_witness_sampling and n_points > witness_threshold:
        max_witnesses = witness_config.get('max_witnesses', 60000)
        actual_witnesses = min(max_witnesses, n_points)
        sampling_method = witness_config.get('witness_sampling_method', 'random')
        print(f"   • Witness sampling: ENABLED ({sampling_method})")
        print(f"   • Actual witnesses: {actual_witnesses} (from {n_points} total)")
    else:
        print(f"   • Witness sampling: DISABLED")
        print(f"   • Using all witnesses: {n_points}")
    
    print(f"   • Relaxation (ν): {witness_config['relaxation']}")
    print(f"   • Max α²: {witness_config['max_alpha_square']}")
    print(f"   • Witness type: {witness_config['witness_type']}")
    print(f"   • Max dimension: {computation_config['max_dimension']}")
    
    try:
        print(f"\n🔄 Starting witness complex computation...")
        print(f"   • Algorithm will use {actual_landmarks} landmarks")
        
        # Call the witness complex computation function directly on the data
        # process_single_layer_witness_optimized expects (n_points, n_dims) array
        betti_results = process_single_layer_witness_optimized(
            X, 
            homology_config,
            layer_idx=0
        )
        
        computation_time = time.time() - start_time
        
        # betti_results is a list of Betti numbers [β0, β1, β2, ...]
        if betti_results is not None and len(betti_results) > 0:
            layer_betti = np.array(betti_results)  # Convert to numpy array
            
            print(f"\n✅ COMPUTATION COMPLETED")
            print("=" * 30)
            print(f"⏱️  Total time: {computation_time:.2f}s")
            print(f"📈 Betti numbers:")
            for i, betti in enumerate(layer_betti):
                print(f"   • β₊{i}: {int(betti)}")
            
            # Create results dictionary
            results = {
                'betti_numbers': layer_betti,
                'computation_time': computation_time,
                'config_used': homology_config,
                'dataset_shape': X.shape,
                'method': 'witness_complex',
                'parallel': use_parallel,
                'success': True
            }
            
            return results
        else:
            print(f"\n❌ COMPUTATION FAILED")
            print("   No valid Betti numbers computed")
            return {
                'betti_numbers': None,
                'computation_time': computation_time,
                'success': False,
                'error': 'No results returned from witness complex computation'
            }
            
    except Exception as e:
        computation_time = time.time() - start_time
        print(f"\n❌ COMPUTATION FAILED")
        print(f"   Error: {e}")
        print(f"   Time elapsed: {computation_time:.2f}s")
        
        return {
            'betti_numbers': None,
            'computation_time': computation_time,
            'success': False,
            'error': str(e)
        }


def save_results(results: Dict, homology_config: Dict):
    """Save computation results to files."""
    if not results['success']:
        print("⚠️  Skipping save due to computation failure")
        return
    
    output_config = homology_config['output']
    output_dir = Path(output_config.get('output_dir', 'results/homology'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n💾 SAVING RESULTS")
    print("-" * 20)
    
    if output_config.get('save_betti', True):
        # Save Betti numbers
        betti_file = output_dir / f"torus_betti_numbers_witness_{timestamp}.pt"
        
        # Convert to format compatible with other scripts
        import torch
        betti_tensor = torch.tensor(results['betti_numbers']).unsqueeze(0).unsqueeze(0)  # [1, 1, max_dim+1]
        
        torch.save({
            'betti_numbers': betti_tensor,
            'metadata': {
                'method': 'witness_complex',
                'computation_time': results['computation_time'],
                'dataset_shape': results['dataset_shape'],
                'config': results['config_used'],
                'timestamp': timestamp
            }
        }, betti_file)
        
        print(f"   ✅ Betti numbers: {betti_file}")
    
    # Save detailed results as YAML for human readability
    results_file = output_dir / f"torus_homology_witness_results_{timestamp}.yaml"
    
    # Prepare data for YAML (convert numpy arrays to lists)
    yaml_results = {
        'computation_summary': {
            'method': 'witness_complex',
            'success': results['success'],
            'computation_time_seconds': float(results['computation_time']),
            'dataset_shape': list(results['dataset_shape']),
            'timestamp': timestamp
        },
        'betti_numbers': {
            f'beta_{i}': int(betti) for i, betti in enumerate(results['betti_numbers'])
        },
        'configuration_used': results['config_used']
    }
    
    with open(results_file, 'w') as f:
        yaml.dump(yaml_results, f, default_flow_style=False, indent=2)
    
    print(f"   ✅ Detailed results: {results_file}")
    print(f"   📁 Output directory: {output_dir}")


def print_comparison_info(results: Dict):
    """Print basic results summary."""
    if not results['success']:
        return
    
    betti = results['betti_numbers']
    print(f"\n🎯 Final Results:")
    for i, b in enumerate(betti):
        print(f"   β₊{i} = {int(b)}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute homology of torus dataset using witness complexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/utils/compute_torus_homology_witness.py
  python src/utils/compute_torus_homology_witness.py --parallel
  python src/utils/compute_torus_homology_witness.py --no-parallel --config custom_config.yaml
        """
    )
    
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Use parallel processing (default: True)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--config', type=str,
                       help='Path to custom homology configuration file')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results to files (default: True)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    # Handle parallel processing flags
    use_parallel = args.parallel and not args.no_parallel
    save_results_flag = args.save and not args.no_save
    
    print("🧮 TORUS HOMOLOGY COMPUTATION WITH WITNESS COMPLEXES")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️  Parallel processing: {'Enabled' if use_parallel else 'Disabled'}")
    print(f"💾 Save results: {'Enabled' if save_results_flag else 'Disabled'}")
    
    try:
        # Load configurations
        print(f"\n📋 Loading configurations...")
        training_config, homology_config = load_configs()
        
        # Override with custom config if provided
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    custom_config = yaml.safe_load(f)
                homology_config.update(custom_config)
                print(f"   ✅ Loaded custom config: {config_path}")
            else:
                print(f"   ⚠️  Custom config not found: {config_path}, using default")
        
        # Generate dataset
        print(f"\n🎲 Dataset generation...")
        X, y = generate_torus_dataset(training_config)
        
        # Prepare dataset for witness complex computation
        print(f"\n🎯 Dataset preparation...")
        X_prepared = prepare_dataset_for_witness(X, homology_config)
        
        # Compute homology
        results = compute_witness_homology(X_prepared, homology_config, use_parallel)
        
        # Save results
        if save_results_flag and results['success']:
            save_results(results, homology_config)
        
        # Print comparison information
        print_comparison_info(results)
        
        # Final summary
        total_time = results['computation_time']
        if results['success']:
            print(f"\n🎉 SUCCESS! Completed in {total_time:.2f} seconds")
            return 0
        else:
            print(f"\n💥 FAILED after {total_time:.2f} seconds")
            print(f"   Error: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Computation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        print(f"\n🕐 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    exit(main())