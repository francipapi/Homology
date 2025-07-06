"""
MNIST Image PCA Preprocessing

This module creates lower resolution MNIST images using PCA to reduce spatial dimensions
while preserving the 2D image structure for CNN training.
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
import pickle
import matplotlib.pyplot as plt


class MNISTImagePCA:
    """
    MNIST preprocessor that creates lower resolution images using PCA
    while maintaining 2D image structure for CNN training.
    """
    
    def __init__(self, 
                 data_root: str = "data/mnist",
                 cache_dir: str = "data/processed",
                 random_seed: int = 42):
        """
        Initialize the MNIST Image PCA processor.
        
        Args:
            data_root: Directory to store raw MNIST data
            cache_dir: Directory to store processed data
            random_seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir)
        self.random_seed = random_seed
        
        # Create directories if they don't exist
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def load_mnist_raw(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load raw MNIST data.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Define transform to convert PIL Image to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Download and load MNIST
        train_dataset = torchvision.datasets.MNIST(
            root=str(self.data_root), 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root=str(self.data_root), 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # Convert to numpy arrays
        X_train = train_dataset.data.numpy().astype(np.float32) / 255.0
        y_train = train_dataset.targets.numpy().astype(np.int64)
        
        X_test = test_dataset.data.numpy().astype(np.float32) / 255.0
        y_test = test_dataset.targets.numpy().astype(np.int64)
        
        print(f"Loaded MNIST: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, y_train, X_test, y_test
    
    def create_pca_images(self, 
                         X_train: np.ndarray, 
                         X_test: np.ndarray,
                         target_resolution: int = 14,
                         preserve_spatial: bool = True) -> Tuple[np.ndarray, np.ndarray, PCA]:
        """
        Create lower resolution images using PCA.
        
        Args:
            X_train: Training images (N, 28, 28)
            X_test: Test images (N, 28, 28)
            target_resolution: Target image size (e.g., 14 for 14x14 images)
            preserve_spatial: If True, reshape back to 2D images
            
        Returns:
            Tuple of (X_train_pca, X_test_pca, pca_model)
        """
        print(f"Creating PCA images with target resolution {target_resolution}x{target_resolution}")
        
        # Flatten images for PCA
        X_train_flat = X_train.reshape(X_train.shape[0], -1)  # (N, 784)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)    # (N, 784)
        
        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        
        # Calculate number of PCA components for target resolution
        n_components = target_resolution * target_resolution
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=self.random_seed)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        print(f"Shape after PCA: {X_train_pca.shape}")
        
        if preserve_spatial:
            # Reshape back to 2D images
            X_train_pca = X_train_pca.reshape(-1, target_resolution, target_resolution)
            X_test_pca = X_test_pca.reshape(-1, target_resolution, target_resolution)
            print(f"Reshaped to images: Train {X_train_pca.shape}, Test {X_test_pca.shape}")
        
        return X_train_pca, X_test_pca, pca
    
    def prepare_datasets(self, 
                        target_resolutions: list = [14, 10, 7],
                        binary_classification: bool = False,
                        save_visualizations: bool = True) -> Dict[str, str]:
        """
        Prepare MNIST datasets with different PCA-based resolutions.
        
        Args:
            target_resolutions: List of target image resolutions
            binary_classification: If True, create binary classification (0-4 vs 5-9)
            save_visualizations: If True, save sample visualizations
            
        Returns:
            Dictionary mapping dataset names to file paths
        """
        print("🔄 Creating PCA-based lower resolution MNIST datasets...")
        
        # Load raw MNIST data
        X_train, y_train, X_test, y_test = self.load_mnist_raw()
        
        # Combine train and test for consistent processing
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        
        dataset_paths = {}
        
        for resolution in target_resolutions:
            print(f"\n📐 Creating {resolution}x{resolution} resolution datasets...")
            
            # Create PCA images
            X_all_pca, _, pca_model = self.create_pca_images(
                X_all, np.zeros((1, 28, 28)),  # Dummy test set for unified processing
                target_resolution=resolution
            )
            X_all_pca = X_all_pca[:-1]  # Remove dummy sample
            
            # Process for both multiclass and binary
            for is_binary in [False, True] if binary_classification else [False]:
                if is_binary:
                    # Binary classification: 0-4 vs 5-9
                    y_processed = (y_all >= 5).astype(np.float32)
                    task_name = "binary"
                    output_dim = 1
                else:
                    # Multi-class classification
                    y_processed = y_all.astype(np.int64)
                    task_name = "multiclass"
                    output_dim = 10
                
                # Save dataset
                filename = f"mnist_pca_{resolution}x{resolution}_{task_name}.npy"
                filepath = self.cache_dir / filename
                
                dataset = {
                    'X': X_all_pca.astype(np.float32),
                    'y': y_processed,
                    'resolution': resolution,
                    'task': task_name,
                    'input_shape': [1, resolution, resolution],  # For CNN
                    'output_dim': output_dim,
                    'pca_components': resolution * resolution,
                    'explained_variance': float(pca_model.explained_variance_ratio_.sum())
                }
                
                np.save(filepath, dataset)
                dataset_paths[f"{resolution}x{resolution}_{task_name}"] = str(filepath)
                
                print(f"✅ Saved {filename}")
                print(f"   Shape: {X_all_pca.shape}, Labels: {y_processed.shape}")
                print(f"   Explained variance: {dataset['explained_variance']:.4f}")
                
                # Save visualizations
                if save_visualizations:
                    self._save_visualization(X_all_pca[:16], y_processed[:16], 
                                           resolution, task_name)
        
        return dataset_paths
    
    def _save_visualization(self, 
                           X_samples: np.ndarray, 
                           y_samples: np.ndarray,
                           resolution: int,
                           task_name: str):
        """Save visualization of PCA-processed images."""
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(f'MNIST PCA {resolution}x{resolution} - {task_name}')
        
        for i, ax in enumerate(axes.flat):
            if i < len(X_samples):
                ax.imshow(X_samples[i], cmap='gray')
                ax.set_title(f'Label: {y_samples[i]}')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        viz_path = self.cache_dir / f"mnist_pca_{resolution}x{resolution}_{task_name}_samples.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Visualization saved: {viz_path}")


def main():
    """Main function to create all PCA-based MNIST datasets."""
    processor = MNISTImagePCA()
    
    # Create datasets with different resolutions
    dataset_paths = processor.prepare_datasets(
        target_resolutions=[14, 10, 7],  # 14x14, 10x10, 7x7 images
        binary_classification=True,      # Create both binary and multiclass
        save_visualizations=True
    )
    
    print("\n" + "="*80)
    print("MNIST PCA IMAGE PROCESSING COMPLETE")
    print("="*80)
    
    for name, path in dataset_paths.items():
        print(f"📁 {name}: {path}")
    
    print("\n📋 Usage in training_config.yaml:")
    print("data:")
    print("  type: 'file'")
    print("  data_source: 'data/processed/mnist_pca_14x14_multiclass.npy'")
    print("model:")
    print("  input_dim: 196  # 14*14 for flattened, or use custom_architecture")
    print("custom_architecture:")
    print("  enabled: true")
    print("  input_shape: [1, 14, 14]  # For CNN with 14x14 images")


if __name__ == "__main__":
    main()