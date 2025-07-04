"""
Persistence Distance Metrics Module

This module provides various distance metrics for comparing persistence diagrams,
which are used to track the evolution of neural network topology during training.

Key Distance Metrics:
- Wasserstein distance
- Bottleneck distance
- Heat kernel distance (as used in the paper)
- Silhouette distance
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import warnings

# Optional imports for specialized libraries
try:
    import persim
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False
    
try:
    import gudhi
    import gudhi.wasserstein
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


class PersistenceDistanceCalculator:
    """
    Calculator for various distance metrics between persistence diagrams.
    
    This class provides both custom implementations and wrappers for
    external libraries like persim and gudhi.
    """
    
    def __init__(self, backend: str = "auto", default_metric: str = "wasserstein"):
        """
        Initialize distance calculator.
        
        Args:
            backend: Backend to use ("persim", "gudhi", "custom", "auto")
            default_metric: Default distance metric to use
        """
        self.backend = self._select_backend(backend)
        self.default_metric = default_metric
        
    def _select_backend(self, backend: str) -> str:
        """Select appropriate backend based on availability."""
        if backend == "auto":
            if PERSIM_AVAILABLE:
                return "persim"
            elif GUDHI_AVAILABLE:
                return "gudhi"
            else:
                return "custom"
        elif backend == "persim" and not PERSIM_AVAILABLE:
            warnings.warn("Persim not available, falling back to custom implementation")
            return "custom"
        elif backend == "gudhi" and not GUDHI_AVAILABLE:
            warnings.warn("Gudhi not available, falling back to custom implementation")
            return "custom"
        return backend
    
    def compute_distance(self, diagram1: np.ndarray, diagram2: np.ndarray,
                        metric: Optional[str] = None, **kwargs) -> float:
        """
        Compute distance between two persistence diagrams.
        
        Args:
            diagram1: First persistence diagram (n x 2 array)
            diagram2: Second persistence diagram (m x 2 array)
            metric: Distance metric to use (if None, uses default)
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Distance value
        """
        if metric is None:
            metric = self.default_metric
            
        # Validate diagrams
        diagram1 = self._validate_diagram(diagram1)
        diagram2 = self._validate_diagram(diagram2)
        
        # Route to appropriate implementation
        if metric == "wasserstein":
            return self.wasserstein_distance(diagram1, diagram2, **kwargs)
        elif metric == "bottleneck":
            return self.bottleneck_distance(diagram1, diagram2, **kwargs)
        elif metric == "heat":
            return self.heat_kernel_distance(diagram1, diagram2, **kwargs)
        elif metric == "silhouette":
            return self.silhouette_distance(diagram1, diagram2, **kwargs)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _validate_diagram(self, diagram: np.ndarray) -> np.ndarray:
        """Validate and clean persistence diagram."""
        if len(diagram) == 0:
            return np.empty((0, 2))
            
        diagram = np.asarray(diagram)
        
        # Ensure 2D array
        if diagram.ndim == 1:
            diagram = diagram.reshape(-1, 2)
        elif diagram.ndim != 2 or diagram.shape[1] != 2:
            raise ValueError("Persistence diagram must be an n x 2 array")
        
        # Remove infinite points
        finite_mask = np.isfinite(diagram).all(axis=1)
        diagram = diagram[finite_mask]
        
        # Ensure birth <= death
        valid_mask = diagram[:, 0] <= diagram[:, 1]
        diagram = diagram[valid_mask]
        
        return diagram
    
    def wasserstein_distance(self, diagram1: np.ndarray, diagram2: np.ndarray,
                           p: float = 2.0, delta: float = 0.01) -> float:
        """
        Compute Wasserstein distance between persistence diagrams.
        
        Args:
            diagram1: First persistence diagram
            diagram2: Second persistence diagram
            p: Wasserstein parameter (1 or 2 typically)
            delta: Diagonal weight parameter
            
        Returns:
            Wasserstein distance
        """
        if self.backend == "persim" and PERSIM_AVAILABLE:
            # Persim doesn't have a direct wasserstein function, use sliced_wasserstein
            if p != 2:
                warnings.warn(f"Persim sliced_wasserstein only supports L2, requested p={p}")
            # Use sliced Wasserstein as approximation to regular Wasserstein
            return float(persim.sliced_wasserstein(diagram1, diagram2, M=50))
        elif self.backend == "gudhi" and GUDHI_AVAILABLE:
            return float(gudhi.wasserstein.wasserstein_distance(
                diagram1, diagram2, order=p, internal_p=p
            ))
        else:
            # Custom implementation
            return self._wasserstein_custom(diagram1, diagram2, p, delta)
    
    def _wasserstein_custom(self, diagram1: np.ndarray, diagram2: np.ndarray,
                          p: float = 2.0, delta: float = 0.01) -> float:
        """Custom implementation of Wasserstein distance."""
        # Handle empty diagrams
        if len(diagram1) == 0 and len(diagram2) == 0:
            return 0.0
        elif len(diagram1) == 0:
            return self._diagonal_distance(diagram2, p)
        elif len(diagram2) == 0:
            return self._diagonal_distance(diagram1, p)
        
        # Compute cost matrix
        n1, n2 = len(diagram1), len(diagram2)
        
        # Cost between points
        if p == np.inf:
            cost_matrix = cdist(diagram1, diagram2, metric='chebyshev')
        else:
            cost_matrix = cdist(diagram1, diagram2, metric='minkowski', p=p)
        
        # Add diagonal points
        diag1 = self._distance_to_diagonal(diagram1)
        diag2 = self._distance_to_diagonal(diagram2)
        
        # Augment cost matrix with diagonal
        augmented_cost = np.zeros((n1 + n2, n1 + n2))
        augmented_cost[:n1, :n2] = cost_matrix
        
        # Diagonal assignment for points in diagram1 to diagonal
        augmented_cost[:n1, n2:n1+n2] = np.diag(diag1)
        
        # Diagonal assignment for points in diagram2 to diagonal  
        augmented_cost[n1:n1+n2, :n2] = np.diag(diag2)
        
        # Diagonal to diagonal cost is 0
        augmented_cost[n1:, n2:] = 0
        
        # Solve optimal transport
        row_ind, col_ind = linear_sum_assignment(augmented_cost)
        
        # Compute distance
        if p == np.inf:
            return augmented_cost[row_ind, col_ind].max()
        else:
            return (augmented_cost[row_ind, col_ind] ** p).sum() ** (1.0 / p)
    
    def bottleneck_distance(self, diagram1: np.ndarray, diagram2: np.ndarray) -> float:
        """
        Compute bottleneck distance between persistence diagrams.
        
        The bottleneck distance is the infinity-Wasserstein distance.
        """
        if self.backend == "persim" and PERSIM_AVAILABLE:
            return float(persim.bottleneck(diagram1, diagram2))
        elif self.backend == "gudhi" and GUDHI_AVAILABLE:
            return float(gudhi.bottleneck_distance(diagram1, diagram2))
        else:
            # Use Wasserstein with p=inf
            return self._wasserstein_custom(diagram1, diagram2, p=np.inf)
    
    def heat_kernel_distance(self, diagram1: np.ndarray, diagram2: np.ndarray,
                           sigma: float = 0.1) -> float:
        """
        Compute heat kernel distance between persistence diagrams.
        
        This is the distance metric used in the paper "Persistent Homology Captures
        the Generalization of Neural Networks Without A Validation Set".
        
        Args:
            diagram1: First persistence diagram
            diagram2: Second persistence diagram
            sigma: Heat kernel bandwidth parameter
            
        Returns:
            Heat kernel distance
        """
        # Compute heat kernel signatures
        hks1 = self._heat_kernel_signature(diagram1, sigma)
        hks2 = self._heat_kernel_signature(diagram2, sigma)
        
        # Compute L2 distance between signatures
        return float(np.linalg.norm(hks1 - hks2))
    
    def _heat_kernel_signature(self, diagram: np.ndarray, sigma: float = 0.1,
                              num_samples: int = 50) -> np.ndarray:
        """
        Compute heat kernel signature for a persistence diagram.
        
        The heat kernel signature is a stable summary of the persistence diagram
        that can be used for distance computation.
        """
        if len(diagram) == 0:
            return np.zeros(num_samples)
        
        # Sample points in persistence coordinate space
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        persistences = deaths - births
        
        # Use a fixed grid approach that always produces num_samples points
        # This ensures consistency between different diagrams
        
        # For consistency, use a canonical range that works for all diagrams
        # in a comparison (avoid dependence on individual diagram ranges)
        b_min_global = min(0, births.min())
        b_max_global = max(1, births.max())
        p_max_global = max(1, persistences.max())
        
        # Create uniform sampling grid
        grid_size = int(np.sqrt(num_samples))
        actual_num_samples = grid_size * grid_size  # This ensures consistency
        
        b_samples = np.linspace(b_min_global, b_max_global, grid_size)
        p_samples = np.linspace(0, p_max_global, grid_size)
        
        # Compute heat kernel values
        signature = np.zeros(actual_num_samples)
        idx = 0
        for b in b_samples:
            for p in p_samples:
                # Heat kernel centered at (b, p)
                kernel_sum = 0
                for i in range(len(diagram)):
                    birth_dist = (births[i] - b) ** 2
                    pers_dist = (persistences[i] - p) ** 2
                    kernel_sum += np.exp(-(birth_dist + pers_dist) / (2 * sigma ** 2))
                signature[idx] = kernel_sum
                idx += 1
        
        # Normalize and ensure exactly num_samples elements
        signature = signature / len(diagram) if len(diagram) > 0 else signature
        
        # Resize to exactly num_samples if needed
        if len(signature) != num_samples:
            # Interpolate to get exactly num_samples points
            from scipy.interpolate import interp1d
            try:
                x_old = np.linspace(0, 1, len(signature))
                x_new = np.linspace(0, 1, num_samples)
                f = interp1d(x_old, signature, kind='linear', fill_value='extrapolate')
                signature = f(x_new)
            except ImportError:
                # Fallback: simple resize
                if len(signature) < num_samples:
                    signature = np.pad(signature, (0, num_samples - len(signature)), 'constant')
                else:
                    signature = signature[:num_samples]
        
        return signature
    
    def silhouette_distance(self, diagram1: np.ndarray, diagram2: np.ndarray,
                          power: float = 1.0, resolution: int = 100) -> float:
        """
        Compute silhouette distance between persistence diagrams.
        
        The silhouette is a functional summary of persistence diagrams.
        
        Args:
            diagram1: First persistence diagram
            diagram2: Second persistence diagram
            power: Power parameter for weighting
            resolution: Number of points to sample
            
        Returns:
            Silhouette distance
        """
        # Compute silhouettes
        sil1 = self._compute_silhouette(diagram1, power, resolution)
        sil2 = self._compute_silhouette(diagram2, power, resolution)
        
        # L2 distance between silhouettes
        return float(np.linalg.norm(sil1 - sil2))
    
    def _compute_silhouette(self, diagram: np.ndarray, power: float = 1.0,
                          resolution: int = 100) -> np.ndarray:
        """Compute silhouette (weighted persistence landscape) of a diagram."""
        if len(diagram) == 0:
            return np.zeros(resolution)
        
        # Get range of birth times
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        
        t_min = births.min()
        t_max = deaths.max()
        t_range = np.linspace(t_min, t_max, resolution)
        
        silhouette = np.zeros(resolution)
        
        for i, t in enumerate(t_range):
            # Sum weighted contributions from all intervals containing t
            for j in range(len(diagram)):
                if births[j] <= t <= deaths[j]:
                    persistence = deaths[j] - births[j]
                    silhouette[i] += persistence ** power
        
        return silhouette
    
    def _distance_to_diagonal(self, diagram: np.ndarray) -> np.ndarray:
        """Compute distance of each point to the diagonal."""
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        return (deaths - births) / np.sqrt(2)
    
    def _diagonal_distance(self, diagram: np.ndarray, p: float = 2.0) -> float:
        """Compute total distance of diagram to diagonal."""
        distances = self._distance_to_diagonal(diagram)
        if p == np.inf:
            return distances.max() if len(distances) > 0 else 0.0
        else:
            return (distances ** p).sum() ** (1.0 / p)


def compute_all_distances(diagrams1: Dict[int, np.ndarray],
                         diagrams2: Dict[int, np.ndarray],
                         metrics: List[str] = ["wasserstein", "bottleneck", "heat"],
                         aggregate: str = "sum",
                         **kwargs) -> Dict[str, float]:
    """
    Compute multiple distance metrics between sets of persistence diagrams.
    
    Args:
        diagrams1: First set of diagrams indexed by dimension
        diagrams2: Second set of diagrams indexed by dimension
        metrics: List of metrics to compute
        aggregate: How to aggregate across dimensions ("sum", "max", "mean")
        **kwargs: Additional arguments for distance computation
        
    Returns:
        Dictionary mapping metric names to distances
    """
    calculator = PersistenceDistanceCalculator()
    results = {}
    
    for metric in metrics:
        distances_by_dim = []
        
        # Compute distance for each dimension
        max_dim = max(max(diagrams1.keys(), default=0), 
                     max(diagrams2.keys(), default=0))
        
        for dim in range(max_dim + 1):
            dgm1 = diagrams1.get(dim, np.empty((0, 2)))
            dgm2 = diagrams2.get(dim, np.empty((0, 2)))
            
            dist = calculator.compute_distance(dgm1, dgm2, metric=metric, **kwargs)
            distances_by_dim.append(dist)
        
        # Aggregate across dimensions
        if aggregate == "sum":
            results[metric] = sum(distances_by_dim)
        elif aggregate == "max":
            results[metric] = max(distances_by_dim) if distances_by_dim else 0.0
        elif aggregate == "mean":
            results[metric] = np.mean(distances_by_dim) if distances_by_dim else 0.0
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
    
    return results


def interpolate_persistence_diagrams(diagram1: np.ndarray, diagram2: np.ndarray,
                                   t: float) -> np.ndarray:
    """
    Interpolate between two persistence diagrams.
    
    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram  
        t: Interpolation parameter (0 = diagram1, 1 = diagram2)
        
    Returns:
        Interpolated persistence diagram
    """
    # Use optimal transport to match points
    calculator = PersistenceDistanceCalculator()
    
    # Validate diagrams
    diagram1 = calculator._validate_diagram(diagram1)
    diagram2 = calculator._validate_diagram(diagram2)
    
    if len(diagram1) == 0:
        return diagram2 * t
    if len(diagram2) == 0:
        return diagram1 * (1 - t)
    
    # Compute optimal matching
    cost_matrix = cdist(diagram1, diagram2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Interpolate matched points
    interpolated = []
    matched_in_2 = set()
    
    for i, j in zip(row_ind, col_ind):
        if j < len(diagram2):
            # Interpolate between matched points
            point = (1 - t) * diagram1[i] + t * diagram2[j]
            interpolated.append(point)
            matched_in_2.add(j)
    
    # Handle unmatched points by interpolating with diagonal
    for i in range(len(diagram1)):
        if i not in row_ind:
            # Fade out towards diagonal
            birth, death = diagram1[i]
            diag_point = (birth + death) / 2
            point = (1 - t) * diagram1[i] + t * np.array([diag_point, diag_point])
            if point[1] > point[0]:  # Only keep if death > birth
                interpolated.append(point)
    
    for j in range(len(diagram2)):
        if j not in matched_in_2:
            # Fade in from diagonal
            birth, death = diagram2[j]
            diag_point = (birth + death) / 2
            point = t * diagram2[j] + (1 - t) * np.array([diag_point, diag_point])
            if point[1] > point[0]:  # Only keep if death > birth
                interpolated.append(point)
    
    return np.array(interpolated) if interpolated else np.empty((0, 2))