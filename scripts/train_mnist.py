#!/usr/bin/env python3
"""
MNIST Training Example Script

This script demonstrates how to train neural networks on the MNIST dataset
using the homology analysis pipeline. It supports different PCA dimensions
and both binary and multi-class classification.

Example usage:
    python scripts/train_mnist.py --pca-dim 50 --binary
    python scripts/train_mnist.py --pca-dim 100 --multiclass
    python scripts/train_mnist.py --full-resolution --multiclass
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.mnist_loader import MNISTLoader
from src.models.torch_mlp import train_model as train_mlp
from src.models.torch_custom import train_model as train_custom


def create_mnist_config(base_config_path: str,
                       dataset_path: str,
                       input_dim: int,
                       output_dim: int,
                       loss_fn: str,
                       output_activation: str,
                       architecture_type: str = 'mlp') -> dict:
    """
    Create MNIST-specific configuration based on base config.
    
    Args:
        base_config_path: Path to base training configuration
        dataset_path: Path to prepared MNIST dataset
        input_dim: Input dimension (PCA components or 784)
        output_dim: Output dimension (1 for binary, 10 for multiclass)
        loss_fn: Loss function ('bce' or 'cross_entropy')
        output_activation: Output activation ('sigmoid' or 'softmax')
        architecture_type: Architecture type ('mlp' or 'custom')
        
    Returns:
        Modified configuration dictionary
    """
    # Load base configuration
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model configuration for MNIST
    config['model']['input_dim'] = input_dim
    config['model']['output_dim'] = output_dim
    config['model']['output_activation_fn_name'] = output_activation
    
    # Update training configuration
    config['training']['loss_fn'] = loss_fn
    config['training']['epochs'] = 50  # Increase epochs for MNIST
    config['training']['batch_size'] = 128  # Good batch size for MNIST
    config['training']['learning_rate'] = 0.001  # Standard learning rate for MNIST
    
    # Update data configuration
    config['data']['type'] = 'file'
    config['data']['data_source'] = str(dataset_path)
    
    # Update custom architecture if using custom
    if architecture_type == 'custom':
        config['custom_architecture']['enabled'] = True
        config['custom_architecture']['input_shape'] = [input_dim]
        
        # Create appropriate architecture for MNIST
        if input_dim <= 100:
            # For PCA-reduced data, use simpler MLP
            config['custom_architecture']['layers'] = [
                {'type': 'linear', 'out_features': 128, 'activation': 'relu', 'dropout': 0.2},
                {'type': 'linear', 'out_features': 64, 'activation': 'relu', 'dropout': 0.2},
                {'type': 'linear', 'out_features': output_dim, 'activation': output_activation}
            ]
        else:
            # For full resolution, use deeper network
            config['custom_architecture']['layers'] = [
                {'type': 'linear', 'out_features': 512, 'activation': 'relu', 'dropout': 0.3},
                {'type': 'linear', 'out_features': 256, 'activation': 'relu', 'dropout': 0.3},
                {'type': 'linear', 'out_features': 128, 'activation': 'relu', 'dropout': 0.2},
                {'type': 'linear', 'out_features': output_dim, 'activation': output_activation}
            ]
    else:
        config['custom_architecture']['enabled'] = False
        
        # Update standard MLP configuration
        if input_dim <= 100:
            config['model']['num_hidden_layers'] = 3
            config['model']['hidden_dim'] = 64
        else:
            config['model']['num_hidden_layers'] = 4
            config['model']['hidden_dim'] = 128
    
    return config


def prepare_mnist_datasets(pca_dim: int = None, 
                         binary_classification: bool = False,
                         force_reprocess: bool = False) -> Path:
    """
    Prepare MNIST dataset with specified configuration.
    
    Args:
        pca_dim: PCA dimensions (None for full resolution)
        binary_classification: Whether to create binary classification problem
        force_reprocess: Force reprocessing even if cached
        
    Returns:
        Path to prepared dataset
    """
    print("="*80)
    print("PREPARING MNIST DATASET")
    print("="*80)
    
    loader = MNISTLoader()
    
    dataset_path = loader.prepare_mnist_dataset(
        n_components=pca_dim,
        binary_classification=binary_classification,
        force_reprocess=force_reprocess
    )
    
    return dataset_path


def train_mnist_model(dataset_path: Path,
                     input_dim: int,
                     binary_classification: bool,
                     architecture_type: str = 'mlp',
                     base_config_path: str = 'configs/training_config.yaml') -> None:
    """
    Train neural network on MNIST dataset.
    
    Args:
        dataset_path: Path to prepared MNIST dataset
        input_dim: Input dimension
        binary_classification: Whether using binary classification
        architecture_type: Architecture type ('mlp' or 'custom')
        base_config_path: Path to base training configuration
    """
    print("\n" + "="*80)
    print("TRAINING NEURAL NETWORK ON MNIST")
    print("="*80)
    
    # Determine output configuration
    if binary_classification:
        output_dim = 1
        loss_fn = 'bce'
        output_activation = 'sigmoid'
        task_type = 'binary classification'
    else:
        output_dim = 10
        loss_fn = 'cross_entropy'
        output_activation = 'softmax'
        task_type = 'multiclass classification'
    
    print(f"Task: MNIST {task_type}")
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Architecture: {architecture_type}")
    print(f"Dataset: {dataset_path}")
    
    # Create MNIST-specific configuration
    config = create_mnist_config(
        base_config_path=base_config_path,
        dataset_path=dataset_path,
        input_dim=input_dim,
        output_dim=output_dim,
        loss_fn=loss_fn,
        output_activation=output_activation,
        architecture_type=architecture_type
    )
    
    # Save temporary config
    temp_config_path = Path('temp_mnist_config.yaml')
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Train the model
        if architecture_type == 'custom':
            print(f"\nStarting training with custom architecture...")
            train_custom(str(temp_config_path))
        else:
            print(f"\nStarting training with standard MLP...")
            train_mlp(str(temp_config_path))
            
        print(f"\n✅ Training completed successfully!")
        
    finally:
        # Clean up temporary config
        if temp_config_path.exists():
            temp_config_path.unlink()


def main():
    """
    Main function to handle command line arguments and execute training.
    """
    parser = argparse.ArgumentParser(
        description="Train neural networks on MNIST dataset with homology analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Binary classification with PCA-50
  python scripts/train_mnist.py --pca-dim 50 --binary
  
  # Multiclass classification with PCA-100  
  python scripts/train_mnist.py --pca-dim 100 --multiclass
  
  # Full resolution multiclass with custom architecture
  python scripts/train_mnist.py --full-resolution --multiclass --custom
  
  # Quick test with small PCA
  python scripts/train_mnist.py --pca-dim 20 --binary --quick
        """
    )
    
    # Dataset options
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--pca-dim', type=int, 
                             help='PCA dimensions (e.g., 20, 50, 100, 200)')
    dataset_group.add_argument('--full-resolution', action='store_true',
                             help='Use full 784D MNIST resolution')
    
    # Classification type
    classification_group = parser.add_mutually_exclusive_group(required=True)
    classification_group.add_argument('--binary', action='store_true',
                                   help='Binary classification (0-4 vs 5-9)')
    classification_group.add_argument('--multiclass', action='store_true',
                                    help='10-class classification (0-9)')
    
    # Architecture options
    parser.add_argument('--custom', action='store_true',
                       help='Use custom architecture instead of standard MLP')
    
    # Processing options
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of dataset even if cached')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with reduced parameters for testing')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Base training configuration file')
    
    args = parser.parse_args()
    
    # Determine input dimension
    if args.pca_dim:
        input_dim = args.pca_dim
        pca_dim = args.pca_dim
    else:
        input_dim = 784
        pca_dim = None
    
    # Quick mode adjustments
    if args.quick:
        print("🚀 Quick mode enabled - using reduced parameters for testing")
        if pca_dim and pca_dim > 50:
            print(f"  Reducing PCA dimensions from {pca_dim} to 50")
            pca_dim = 50
            input_dim = 50
    
    # Validate input dimension
    if pca_dim and pca_dim < 10:
        print("⚠️  Warning: Very low PCA dimensions may hurt performance")
    
    try:
        # Step 1: Prepare dataset
        dataset_path = prepare_mnist_datasets(
            pca_dim=pca_dim,
            binary_classification=args.binary,
            force_reprocess=args.force_reprocess
        )
        
        # Step 2: Train model
        architecture_type = 'custom' if args.custom else 'mlp'
        train_mnist_model(
            dataset_path=dataset_path,
            input_dim=input_dim,
            binary_classification=args.binary,
            architecture_type=architecture_type,
            base_config_path=args.config
        )
        
        print("\n" + "="*80)
        print("MNIST TRAINING COMPLETE")
        print("="*80)
        print(f"✅ Successfully trained on MNIST dataset")
        print(f"📊 Input dimension: {input_dim}")
        print(f"🎯 Task: {'Binary' if args.binary else 'Multiclass'} classification")
        print(f"🏗️  Architecture: {architecture_type.upper()}")
        print(f"💾 Dataset: {dataset_path}")
        
        if not args.quick:
            print(f"\n💡 Next steps:")
            print(f"  - Check results in results/ directory")
            print(f"  - Run homology analysis on trained model")
            print(f"  - Try different architectures or PCA dimensions")
            print(f"  - Compare with other models using network_homology_comparison.py")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()