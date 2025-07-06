"""
Data Preprocessing Utilities

This module provides general-purpose data preprocessing utilities for various datasets
including normalization, PCA reduction, format conversion, and validation functions.
These utilities are designed to work with the existing training pipeline.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union, List
import pickle
import warnings


class DataPreprocessor:
    """
    General-purpose data preprocessor with support for various transformations.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.fitted_transformers = {}
        np.random.seed(random_seed)
    
    def standardize(self, 
                   X_train: np.ndarray, 
                   X_test: Optional[np.ndarray] = None,
                   scaler_type: str = 'standard') -> Tuple[np.ndarray, Optional[np.ndarray], Any]:
        """
        Standardize data using various scaling methods.
        
        Args:
            X_train: Training data
            X_test: Test data (optional)
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, scaler)
        """
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
        # Fit on training data
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Transform test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
        
        # Store fitted transformer
        self.fitted_transformers['scaler'] = scaler
        
        print(f"Data standardized using {scaler_type} scaler")
        return X_train_scaled, X_test_scaled, scaler
    
    def apply_pca(self, 
                  X_train: np.ndarray, 
                  X_test: Optional[np.ndarray] = None,
                  n_components: int = 50,
                  explained_variance_threshold: Optional[float] = None) -> Tuple[np.ndarray, Optional[np.ndarray], PCA]:
        """
        Apply PCA dimensionality reduction.
        
        Args:
            X_train: Training data
            X_test: Test data (optional)
            n_components: Number of components or float for variance ratio
            explained_variance_threshold: If set, use enough components to explain this variance
            
        Returns:
            Tuple of (X_train_pca, X_test_pca, pca_model)
        """
        # Determine number of components
        if explained_variance_threshold is not None:
            # First fit with all components to determine required number
            temp_pca = PCA(random_state=self.random_seed)
            temp_pca.fit(X_train)
            cumsum_var = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= explained_variance_threshold) + 1
            print(f"Using {n_components} components to explain {explained_variance_threshold:.1%} variance")
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=self.random_seed)
        X_train_pca = pca.fit_transform(X_train)
        
        # Transform test data if provided
        X_test_pca = None
        if X_test is not None:
            X_test_pca = pca.transform(X_test)
        
        # Store fitted transformer
        self.fitted_transformers['pca'] = pca
        
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"PCA applied: {n_components} components, {explained_variance:.2%} variance explained")
        
        return X_train_pca, X_test_pca, pca
    
    def apply_tsne(self, 
                   X: np.ndarray,
                   n_components: int = 2,
                   perplexity: float = 30.0,
                   learning_rate: float = 200.0) -> np.ndarray:
        """
        Apply t-SNE dimensionality reduction for visualization.
        Note: t-SNE doesn't support transform on new data, only fit_transform.
        
        Args:
            X: Input data
            n_components: Number of dimensions for output
            perplexity: t-SNE perplexity parameter
            learning_rate: t-SNE learning rate
            
        Returns:
            X_tsne: Transformed data
        """
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=self.random_seed
        )
        
        X_tsne = tsne.fit_transform(X)
        print(f"t-SNE applied: {X.shape[1]}D -> {n_components}D")
        
        return X_tsne
    
    def create_binary_classification(self, 
                                   y: np.ndarray,
                                   positive_classes: List[int],
                                   negative_classes: Optional[List[int]] = None) -> np.ndarray:
        """
        Convert multi-class labels to binary classification.
        
        Args:
            y: Original labels
            positive_classes: Classes to map to 1
            negative_classes: Classes to map to 0 (if None, use all others)
            
        Returns:
            Binary labels (0 or 1)
        """
        if negative_classes is None:
            # Use all classes not in positive_classes
            unique_classes = np.unique(y)
            negative_classes = [c for c in unique_classes if c not in positive_classes]
        
        y_binary = np.zeros_like(y, dtype=int)
        
        # Set positive classes to 1
        for cls in positive_classes:
            y_binary[y == cls] = 1
        
        # Filter to only include specified classes
        all_classes = positive_classes + negative_classes
        mask = np.isin(y, all_classes)
        
        print(f"Created binary classification: {positive_classes} vs {negative_classes}")
        print(f"Kept {np.sum(mask)} samples out of {len(y)}")
        
        return y_binary[mask], mask
    
    def split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   train_ratio: float = 0.8,
                   stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Labels
            train_ratio: Proportion for training set
            stratify: Whether to stratify split based on labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = 1.0 - train_ratio
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            stratify=stratify_param,
            random_state=self.random_seed
        )
        
        print(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def normalize_images(self, 
                        X: np.ndarray,
                        method: str = 'minmax') -> np.ndarray:
        """
        Normalize image data to [0, 1] or [-1, 1] range.
        
        Args:
            X: Image data (assumed to be in [0, 255] range)
            method: Normalization method ('minmax' for [0,1], 'standard' for [-1,1])
            
        Returns:
            Normalized image data
        """
        if method == 'minmax':
            # Normalize to [0, 1]
            X_norm = X.astype(np.float32) / 255.0
        elif method == 'standard':
            # Normalize to [-1, 1]
            X_norm = (X.astype(np.float32) / 255.0) * 2.0 - 1.0
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        print(f"Images normalized using {method} method")
        return X_norm


class DataValidator:
    """
    Utilities for validating dataset integrity and compatibility.
    """
    
    @staticmethod
    def validate_dataset(X: np.ndarray, 
                        y: np.ndarray,
                        expected_input_dim: Optional[int] = None,
                        expected_output_classes: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate dataset integrity and return summary statistics.
        
        Args:
            X: Feature data
            y: Labels
            expected_input_dim: Expected input dimension
            expected_output_classes: Expected number of output classes
            
        Returns:
            Dictionary with validation results and dataset statistics
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Basic shape validation
        if X.shape[0] != y.shape[0]:
            validation_results['errors'].append(
                f"Shape mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples"
            )
            validation_results['valid'] = False
        
        # Input dimension validation
        if expected_input_dim is not None and X.shape[1] != expected_input_dim:
            validation_results['errors'].append(
                f"Input dimension mismatch: expected {expected_input_dim}, got {X.shape[1]}"
            )
            validation_results['valid'] = False
        
        # Output classes validation
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        if expected_output_classes is not None and n_classes != expected_output_classes:
            validation_results['warnings'].append(
                f"Class count mismatch: expected {expected_output_classes}, found {n_classes}"
            )
        
        # Check for missing values
        if np.any(np.isnan(X)):
            validation_results['errors'].append("NaN values found in features")
            validation_results['valid'] = False
        
        if np.any(np.isnan(y)):
            validation_results['errors'].append("NaN values found in labels")
            validation_results['valid'] = False
        
        # Check for infinite values
        if np.any(np.isinf(X)):
            validation_results['errors'].append("Infinite values found in features")
            validation_results['valid'] = False
        
        # Collect statistics
        validation_results['statistics'] = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': n_classes,
            'class_distribution': {int(cls): int(np.sum(y == cls)) for cls in unique_classes},
            'feature_range': {
                'min': float(np.min(X)),
                'max': float(np.max(X)),
                'mean': float(np.mean(X)),
                'std': float(np.std(X))
            }
        }
        
        return validation_results
    
    @staticmethod
    def print_validation_report(validation_results: Dict[str, Any]) -> None:
        """
        Print a formatted validation report.
        
        Args:
            validation_results: Results from validate_dataset()
        """
        print("\n" + "="*50)
        print("DATASET VALIDATION REPORT")
        print("="*50)
        
        # Overall status
        status = "✅ VALID" if validation_results['valid'] else "❌ INVALID"
        print(f"Status: {status}")
        
        # Errors
        if validation_results['errors']:
            print(f"\n❌ Errors ({len(validation_results['errors'])}):")
            for error in validation_results['errors']:
                print(f"  - {error}")
        
        # Warnings
        if validation_results['warnings']:
            print(f"\n⚠️  Warnings ({len(validation_results['warnings'])}):")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")
        
        # Statistics
        stats = validation_results['statistics']
        print(f"\n📊 Dataset Statistics:")
        print(f"  Samples: {stats['n_samples']:,}")
        print(f"  Features: {stats['n_features']:,}")
        print(f"  Classes: {stats['n_classes']}")
        
        print(f"\n📈 Feature Statistics:")
        print(f"  Range: [{stats['feature_range']['min']:.4f}, {stats['feature_range']['max']:.4f}]")
        print(f"  Mean: {stats['feature_range']['mean']:.4f}")
        print(f"  Std: {stats['feature_range']['std']:.4f}")
        
        print(f"\n🎯 Class Distribution:")
        for cls, count in stats['class_distribution'].items():
            percentage = 100 * count / stats['n_samples']
            print(f"  Class {cls}: {count:,} samples ({percentage:.1f}%)")


class FormatConverter:
    """
    Utilities for converting between different data formats.
    """
    
    @staticmethod
    def save_data(X: np.ndarray, 
                  y: np.ndarray,
                  filepath: Union[str, Path],
                  format: str = 'auto',
                  metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save data in various formats compatible with the training pipeline.
        
        Args:
            X: Feature data
            y: Labels
            filepath: Output file path
            format: Output format ('npy', 'npz', 'pt', 'auto')
            metadata: Optional metadata dictionary
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        
        # Auto-detect format from extension
        if format == 'auto':
            extension = filepath.suffix.lower()
            if extension == '.npy':
                format = 'npy'
            elif extension == '.npz':
                format = 'npz'
            elif extension in ['.pt', '.pth']:
                format = 'pt'
            else:
                raise ValueError(f"Cannot auto-detect format from extension: {extension}")
        
        # Prepare data dictionary (compatible with existing pipeline)
        data_dict = {
            'X': X,
            'y': y,
            'input_dim': X.shape[1],
            'n_samples': X.shape[0],
            'n_classes': len(np.unique(y))
        }
        
        if metadata:
            data_dict['metadata'] = metadata
        
        # Save in requested format
        if format == 'npy':
            np.save(filepath, data_dict)
        elif format == 'npz':
            np.savez_compressed(filepath, **data_dict)
        elif format == 'pt':
            # Convert to PyTorch tensors
            torch_dict = {}
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    torch_dict[key] = torch.from_numpy(value)
                else:
                    torch_dict[key] = value
            torch.save(torch_dict, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data saved to: {filepath}")
        return filepath
    
    @staticmethod
    def load_data(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load data from various formats.
        
        Args:
            filepath: Path to data file
            
        Returns:
            Tuple of (X, y, metadata)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        extension = filepath.suffix.lower()
        
        if extension == '.npy':
            data_dict = np.load(filepath, allow_pickle=True).item()
        elif extension == '.npz':
            data_dict = dict(np.load(filepath))
        elif extension in ['.pt', '.pth']:
            data_dict = torch.load(filepath, map_location='cpu')
            # Convert tensors back to numpy
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.numpy()
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        X = data_dict['X']
        y = data_dict['y']
        metadata = data_dict.get('metadata', {})
        
        print(f"Data loaded from: {filepath}")
        print(f"Shape: X{X.shape}, y{y.shape}")
        
        return X, y, metadata


def main():
    """
    Example usage of preprocessing utilities.
    """
    print("Data Preprocessing Utilities - Example Usage")
    print("="*50)
    
    # Example with synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 784)  # Simulate flattened MNIST
    y = np.random.randint(0, 10, 1000)  # 10 classes
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(random_seed=42)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, train_ratio=0.8)
    
    # Standardize
    X_train_scaled, X_test_scaled, scaler = preprocessor.standardize(X_train, X_test)
    
    # Apply PCA
    X_train_pca, X_test_pca, pca = preprocessor.apply_pca(
        X_train_scaled, X_test_scaled, n_components=50
    )
    
    # Create binary classification
    y_binary, mask = preprocessor.create_binary_classification(
        y_train, positive_classes=[0, 1, 2, 3, 4]
    )
    X_binary = X_train_pca[mask]
    
    # Validate dataset
    validator = DataValidator()
    validation_results = validator.validate_dataset(
        X_binary, y_binary, expected_input_dim=50, expected_output_classes=2
    )
    validator.print_validation_report(validation_results)
    
    # Save data
    converter = FormatConverter()
    output_path = converter.save_data(
        X_binary, y_binary, 'example_dataset.npy',
        metadata={'preprocessing_steps': ['standardization', 'pca_50', 'binary_classification']}
    )
    
    print(f"\n✅ Example preprocessing complete!")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()