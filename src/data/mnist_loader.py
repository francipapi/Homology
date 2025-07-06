"""
Enhanced MNIST Data Loader with PCA Support

This module provides a comprehensive MNIST data loader that integrates with the existing
training pipeline. It supports automatic downloading, preprocessing, PCA dimensionality
reduction, and export to compatible formats.

Features:
- Automatic MNIST download using torchvision
- Configurable PCA dimensionality reduction (50, 100, 200 components)
- Support for both flattened (784D) and PCA-reduced data
- Compatible data format export (.npy, .npz, .pt)
- Multi-class vs binary classification options
- Proper train/test split handling
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


class MNISTLoader:
    """
    Enhanced MNIST data loader with PCA support and pipeline integration.
    """
    
    def __init__(self, 
                 data_root: str = "data/mnist",
                 cache_dir: str = "data/processed",
                 random_seed: int = 42):
        """
        Initialize the MNIST loader.
        
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
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Cache for loaded data
        self._raw_data_cache = {}
        self._processed_data_cache = {}
    
    def download_mnist(self, force_download: bool = False) -> None:
        """
        Download MNIST dataset using torchvision.
        
        Args:
            force_download: Force re-download even if data exists
        """
        print("Downloading MNIST dataset...")
        
        # Define transform to convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Download training and test sets
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
        
        print(f"✅ MNIST downloaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    def load_raw_mnist(self, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load raw MNIST data and return as numpy arrays.
        
        Args:
            normalize: Whether to normalize pixel values to [0, 1]
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        cache_key = f"raw_mnist_norm_{normalize}"
        
        if cache_key in self._raw_data_cache:
            return self._raw_data_cache[cache_key]
        
        print("Loading raw MNIST data...")
        
        # Ensure data is downloaded
        self.download_mnist()
        
        # Load datasets
        transform = transforms.Compose([transforms.ToTensor()])
        
        train_dataset = torchvision.datasets.MNIST(
            root=str(self.data_root),
            train=True,
            download=False,
            transform=transform
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root=str(self.data_root),
            train=False,
            download=False,
            transform=transform
        )
        
        # Convert to numpy arrays
        X_train = []
        y_train = []
        for data, target in train_dataset:
            X_train.append(data.numpy())
            y_train.append(target)
        
        X_test = []
        y_test = []
        for data, target in test_dataset:
            X_test.append(data.numpy())
            y_test.append(target)
        
        # Convert to numpy arrays and reshape
        X_train = np.array(X_train).reshape(-1, 784)  # Flatten to 784D
        X_test = np.array(X_test).reshape(-1, 784)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # Normalize if requested
        if normalize:
            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0
        
        result = (X_train, y_train, X_test, y_test)
        self._raw_data_cache[cache_key] = result
        
        print(f"✅ Raw MNIST loaded: X_train {X_train.shape}, X_test {X_test.shape}")
        return result
    
    def apply_pca(self, 
                  X_train: np.ndarray, 
                  X_test: np.ndarray,
                  n_components: int = 50,
                  standardize: bool = True) -> Tuple[np.ndarray, np.ndarray, PCA, Optional[StandardScaler]]:
        """
        Apply PCA dimensionality reduction to MNIST data.
        
        Args:
            X_train: Training features
            X_test: Test features
            n_components: Number of PCA components to keep
            standardize: Whether to standardize data before PCA
            
        Returns:
            Tuple of (X_train_pca, X_test_pca, pca_model, scaler)
        """
        print(f"Applying PCA with {n_components} components...")
        
        scaler = None
        if standardize:
            print("  Standardizing data...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Apply PCA
        print(f"  Fitting PCA with {n_components} components...")
        pca = PCA(n_components=n_components, random_state=self.random_seed)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Calculate explained variance
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"  ✅ PCA complete: {explained_variance:.2%} variance explained")
        
        return X_train_pca, X_test_pca, pca, scaler
    
    def create_binary_classification(self, 
                                   y_train: np.ndarray, 
                                   y_test: np.ndarray,
                                   positive_classes: list = [0, 1, 2, 3, 4],
                                   negative_classes: list = [5, 6, 7, 8, 9]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert MNIST to binary classification problem.
        
        Args:
            y_train: Training labels
            y_test: Test labels
            positive_classes: List of digits to map to class 1
            negative_classes: List of digits to map to class 0
            
        Returns:
            Tuple of (y_train_binary, y_test_binary)
        """
        print(f"Creating binary classification: {positive_classes} vs {negative_classes}")
        
        # Create binary labels
        y_train_binary = np.zeros_like(y_train)
        y_test_binary = np.zeros_like(y_test)
        
        for cls in positive_classes:
            y_train_binary[y_train == cls] = 1
            y_test_binary[y_test == cls] = 1
        
        # Keep only samples from specified classes
        train_mask = np.isin(y_train, positive_classes + negative_classes)
        test_mask = np.isin(y_test, positive_classes + negative_classes)
        
        y_train_binary = y_train_binary[train_mask]
        y_test_binary = y_test_binary[test_mask]
        
        print(f"  ✅ Binary classification created: {len(y_train_binary)} train, {len(y_test_binary)} test samples")
        return y_train_binary, y_test_binary, train_mask, test_mask
    
    def save_processed_data(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           filename: str,
                           format: str = 'npy',
                           metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save processed data in compatible format.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            filename: Base filename (without extension)
            format: Output format ('npy', 'npz', 'pt')
            metadata: Optional metadata dictionary
            
        Returns:
            Path to saved file
        """
        # Prepare data dictionary
        data_dict = {
            'X': np.concatenate([X_train, X_test], axis=0),
            'y': np.concatenate([y_train, y_test], axis=0),
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'input_dim': X_train.shape[1],
            'num_classes': len(np.unique(np.concatenate([y_train, y_test])))
        }
        
        if metadata:
            data_dict['metadata'] = metadata
        
        # Save in requested format
        if format == 'npy':
            filepath = self.cache_dir / f"{filename}.npy"
            np.save(filepath, data_dict)
        elif format == 'npz':
            filepath = self.cache_dir / f"{filename}.npz"
            np.savez_compressed(filepath, **data_dict)
        elif format == 'pt':
            # Convert to PyTorch tensors
            torch_dict = {}
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    torch_dict[key] = torch.from_numpy(value)
                else:
                    torch_dict[key] = value
            
            filepath = self.cache_dir / f"{filename}.pt"
            torch.save(torch_dict, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ Data saved to: {filepath}")
        return filepath
    
    def prepare_mnist_dataset(self,
                             n_components: Optional[int] = None,
                             binary_classification: bool = False,
                             positive_classes: list = [0, 1, 2, 3, 4],
                             negative_classes: list = [5, 6, 7, 8, 9],
                             standardize: bool = True,
                             output_format: str = 'npy',
                             force_reprocess: bool = False) -> Path:
        """
        Complete MNIST dataset preparation pipeline.
        
        Args:
            n_components: Number of PCA components (None for full 784D)
            binary_classification: Whether to create binary classification problem
            positive_classes: Classes to map to 1 (for binary classification)
            negative_classes: Classes to map to 0 (for binary classification)
            standardize: Whether to standardize data before PCA
            output_format: Output format ('npy', 'npz', 'pt')
            force_reprocess: Force reprocessing even if cached file exists
            
        Returns:
            Path to processed dataset file
        """
        # Generate filename based on configuration
        filename_parts = ["mnist"]
        
        if n_components is not None:
            filename_parts.append(f"pca_{n_components}")
        else:
            filename_parts.append("full")
        
        if binary_classification:
            filename_parts.append("binary")
        else:
            filename_parts.append("multiclass")
        
        if standardize and n_components is not None:
            filename_parts.append("standardized")
        
        filename = "_".join(filename_parts)
        
        # Check if processed file already exists
        output_file = self.cache_dir / f"{filename}.{output_format}"
        if output_file.exists() and not force_reprocess:
            print(f"✅ Using cached dataset: {output_file}")
            return output_file
        
        print(f"🔄 Processing MNIST dataset: {filename}")
        
        # Load raw data
        X_train, y_train, X_test, y_test = self.load_raw_mnist(normalize=True)
        
        # Apply binary classification if requested
        if binary_classification:
            y_train_binary, y_test_binary, train_mask, test_mask = self.create_binary_classification(
                y_train, y_test, positive_classes, negative_classes
            )
            X_train = X_train[train_mask]
            X_test = X_test[test_mask]
            y_train = y_train_binary
            y_test = y_test_binary
        
        # Apply PCA if requested
        if n_components is not None:
            X_train, X_test, pca_model, scaler = self.apply_pca(
                X_train, X_test, n_components, standardize
            )
        else:
            pca_model = None
            scaler = None
        
        # Prepare metadata
        metadata = {
            'dataset': 'MNIST',
            'n_components': n_components,
            'binary_classification': binary_classification,
            'positive_classes': positive_classes if binary_classification else None,
            'negative_classes': negative_classes if binary_classification else None,
            'standardize': standardize,
            'random_seed': self.random_seed,
            'original_shape': (28, 28),
            'preprocessing_steps': []
        }
        
        if scaler is not None:
            metadata['preprocessing_steps'].append('standardization')
        if pca_model is not None:
            metadata['preprocessing_steps'].append(f'pca_{n_components}')
            metadata['explained_variance_ratio'] = float(np.sum(pca_model.explained_variance_ratio_))
        
        # Save processed data
        return self.save_processed_data(
            X_train, y_train, X_test, y_test,
            filename, output_format, metadata
        )
    
    def create_all_standard_datasets(self, 
                                   output_format: str = 'npy',
                                   force_reprocess: bool = False) -> Dict[str, Path]:
        """
        Create all standard MNIST dataset variants.
        
        Args:
            output_format: Output format for all datasets
            force_reprocess: Force reprocessing even if cached files exist
            
        Returns:
            Dictionary mapping dataset names to file paths
        """
        datasets = {}
        
        # Standard PCA dimensions
        pca_dimensions = [50, 100, 200]
        
        print("🔄 Creating all standard MNIST datasets...")
        
        # Full resolution datasets
        print("\n1. Creating full resolution datasets...")
        datasets['mnist_full_multiclass'] = self.prepare_mnist_dataset(
            n_components=None,
            binary_classification=False,
            output_format=output_format,
            force_reprocess=force_reprocess
        )
        
        datasets['mnist_full_binary'] = self.prepare_mnist_dataset(
            n_components=None,
            binary_classification=True,
            output_format=output_format,
            force_reprocess=force_reprocess
        )
        
        # PCA-reduced datasets
        for n_comp in pca_dimensions:
            print(f"\n2. Creating PCA-{n_comp} datasets...")
            
            # Multi-class
            datasets[f'mnist_pca_{n_comp}_multiclass'] = self.prepare_mnist_dataset(
                n_components=n_comp,
                binary_classification=False,
                output_format=output_format,
                force_reprocess=force_reprocess
            )
            
            # Binary classification
            datasets[f'mnist_pca_{n_comp}_binary'] = self.prepare_mnist_dataset(
                n_components=n_comp,
                binary_classification=True,
                output_format=output_format,
                force_reprocess=force_reprocess
            )
        
        print(f"\n✅ All datasets created: {len(datasets)} total")
        return datasets


def main():
    """
    Example usage of the MNIST loader.
    """
    # Initialize loader
    loader = MNISTLoader()
    
    # Create all standard datasets
    datasets = loader.create_all_standard_datasets(
        output_format='npy',
        force_reprocess=False
    )
    
    # Print summary
    print("\n" + "="*80)
    print("MNIST DATASET PREPARATION COMPLETE")
    print("="*80)
    
    for name, path in datasets.items():
        print(f"📁 {name}: {path}")
    
    print("\n📋 Usage in training_config.yaml:")
    print("data:")
    print("  type: 'file'")
    print("  data_source: 'data/processed/mnist_pca_50_binary.npy'")
    print("model:")
    print("  input_dim: 50  # or 100, 200, 784")
    print("  output_dim: 2  # or 10 for multiclass")


if __name__ == "__main__":
    main()