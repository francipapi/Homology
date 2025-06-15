"""
Computes topological features using Ripser - a fast implementation of Vietoris-Rips persistent homology.

This module provides a Ripser-based alternative to the Gudhi implementation,
offering significant performance improvements (typically 5-100x faster).
Ripser is particularly optimized for Vietoris-Rips filtrations and uses
various algorithmic improvements including implicit matrix reduction and
cohomology computation.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from ripser import ripser
from persim import plot_diagrams
import warnings


def compute_persistent_homology(distance_matrix, max_dimension=2, max_edge_length=1.0,
                                diag_filename=None,
                                plot_barcode_filename=None,
                                betti_filename=None):
    """
    Computes persistent homology using Ripser for improved performance.
    
    This implementation offers significant speedups compared to Gudhi,
    especially for larger datasets and higher dimensions.
    
    Args:
        distance_matrix (np.ndarray): A square, symmetric 2D NumPy array
            representing the distance matrix of the dataset. Diagonal elements
            must be zero.
        max_dimension (int, optional): The maximum homology dimension to compute
            (e.g., 2 means H0, H1, H2). Defaults to 2.
        max_edge_length (float, optional): The maximum edge length (filtration value)
            for including edges in the Vietoris-Rips complex. Defaults to 1.0.
        diag_filename (str, optional): If provided, the persistence diagram
            (dimension, birth, death) tuples are saved to this text file.
            Defaults to None (no file saved).
        plot_barcode_filename (str, optional): If provided, a barcode plot
            visualizing the persistence intervals is saved to this image file.
            Defaults to None (no plot saved).
        betti_filename (str, optional): If provided, the Betti numbers
            (count of topological holes at each dimension) are saved to this
            text file. Defaults to None (no file saved).
    
    Returns:
        list[int]: A list of Betti numbers for dimensions 0 up to `max_dimension`.
                   For example, [B0, B1, B2] if max_dimension is 2.
    
    Raises:
        ValueError: If `distance_matrix` is not a valid NumPy array (e.g., not
            square, not symmetric, or non-zero diagonal).
    """
    
    # Input validation for the distance matrix
    if not isinstance(distance_matrix, np.ndarray):
        raise ValueError("distance_matrix must be a NumPy array.")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square (n x n) matrix.")
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-6):
        raise ValueError("distance_matrix must be symmetric.")
    if not np.all(np.diag(distance_matrix) < 1e-6):
        raise ValueError("The diagonal of distance_matrix must be all zeros.")
    
    # Compute persistent homology using Ripser
    # Ripser returns a dictionary with 'dgms' (persistence diagrams) and 'cocycles'
    result = ripser(distance_matrix, 
                    maxdim=max_dimension,
                    thresh=max_edge_length,
                    distance_matrix=True)
    
    # Extract persistence diagrams
    diagrams = result['dgms']
    
    # Compute Betti numbers by counting features that persist at the max filtration value
    # We use a small epsilon to avoid numerical edge cases
    epsilon = 1e-10
    betti_numbers = []
    
    for dim in range(max_dimension + 1):
        if dim < len(diagrams):
            diagram = diagrams[dim]
            # Count features that are born before max_edge_length and die after it
            # (or are infinite, indicated by death == np.inf)
            if len(diagram) > 0:
                births = diagram[:, 0]
                deaths = diagram[:, 1]
                # Features that persist at max_edge_length
                persistent_features = np.sum((births <= max_edge_length - epsilon) & 
                                           ((deaths > max_edge_length + epsilon) | 
                                            (deaths == np.inf)))
                betti_numbers.append(int(persistent_features))
            else:
                betti_numbers.append(0)
        else:
            betti_numbers.append(0)
    
    # Save persistence diagram to a file if a filename is provided
    if diag_filename:
        os.makedirs(os.path.dirname(diag_filename), exist_ok=True)
        with open(diag_filename, 'w') as f:
            for dim, diagram in enumerate(diagrams):
                for birth, death in diagram:
                    f.write(f"{dim} {birth} {death}\n")
        print(f"Persistence diagram saved to {diag_filename}")
    
    # Generate and save a barcode plot if a filename is provided
    if plot_barcode_filename:
        os.makedirs(os.path.dirname(plot_barcode_filename), exist_ok=True)
        
        # Create barcode plot using persim
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_diagrams(diagrams, ax=ax, show=False)
        ax.set_title(f"Persistence Barcode (max_edge_length={max_edge_length})")
        ax.set_xlim([0, max_edge_length * 1.1])
        
        plt.tight_layout()
        plt.savefig(plot_barcode_filename, dpi=300)
        plt.close()
        print(f"Barcode plot saved to {plot_barcode_filename}")
    
    # Save Betti numbers to a file if a filename is provided
    if betti_filename:
        os.makedirs(os.path.dirname(betti_filename), exist_ok=True)
        np.savetxt(betti_filename, np.array(betti_numbers), fmt='%d', 
                   header=f"Betti numbers for max_dimension={max_dimension}")
        print(f"Betti numbers saved to {betti_filename}")
    
    return betti_numbers


def compute_persistent_homology_points(points, max_dimension=2, max_edge_length=1.0,
                                      metric='euclidean', n_perm=None):
    """
    Computes persistent homology directly from point cloud data.
    
    This function is more efficient than first computing the full distance matrix,
    especially for large point clouds.
    
    Args:
        points (np.ndarray): An (n, d) array of n points in d dimensions.
        max_dimension (int, optional): Maximum homology dimension. Defaults to 2.
        max_edge_length (float, optional): Maximum filtration value. Defaults to 1.0.
        metric (str, optional): Distance metric to use. Defaults to 'euclidean'.
        n_perm (int, optional): Number of points to subsample. If None, use all points.
    
    Returns:
        dict: Dictionary containing:
            - 'betti_numbers': List of Betti numbers
            - 'diagrams': Persistence diagrams for each dimension
            - 'result': Full Ripser result dictionary
    """
    
    # Subsample if requested
    if n_perm is not None and n_perm < len(points):
        # Use furthest point sampling for better coverage
        indices = fps_sampling(points, n_perm)
        points = points[indices]
    
    # Compute persistent homology
    result = ripser(points,
                    maxdim=max_dimension,
                    thresh=max_edge_length,
                    distance_matrix=False,
                    metric=metric)
    
    diagrams = result['dgms']
    
    # Compute Betti numbers
    epsilon = 1e-10
    betti_numbers = []
    
    for dim in range(max_dimension + 1):
        if dim < len(diagrams):
            diagram = diagrams[dim]
            if len(diagram) > 0:
                births = diagram[:, 0]
                deaths = diagram[:, 1]
                persistent_features = np.sum((births <= max_edge_length - epsilon) & 
                                           ((deaths > max_edge_length + epsilon) | 
                                            (deaths == np.inf)))
                betti_numbers.append(int(persistent_features))
            else:
                betti_numbers.append(0)
        else:
            betti_numbers.append(0)
    
    return {
        'betti_numbers': betti_numbers,
        'diagrams': diagrams,
        'result': result
    }


def fps_sampling(points, n_samples):
    """
    Furthest Point Sampling for subsampling point clouds.
    
    Args:
        points (np.ndarray): (n, d) array of points
        n_samples (int): Number of points to sample
    
    Returns:
        np.ndarray: Indices of sampled points
    """
    n_points = len(points)
    if n_samples >= n_points:
        return np.arange(n_points)
    
    # Start with a random point
    indices = np.zeros(n_samples, dtype=int)
    indices[0] = np.random.randint(n_points)
    
    # Compute distances from the first point
    distances = np.linalg.norm(points - points[indices[0]], axis=1)
    
    for i in range(1, n_samples):
        # Select the farthest point
        indices[i] = np.argmax(distances)
        
        # Update distances
        new_distances = np.linalg.norm(points - points[indices[i]], axis=1)
        distances = np.minimum(distances, new_distances)
    
    return indices


# Parallel processing functions compatible with multiprocessing
def multi_hom(param):
    """
    Computes Betti numbers using Ripser for parallel processing.
    
    Args:
        param (tuple): (distance_matrix, max_edge_length)
    
    Returns:
        list[int]: Betti numbers
    """
    distance_matrix, max_edge_length = param
    max_dimension = 3  # Default, could be passed in param
    
    try:
        result = ripser(distance_matrix,
                       maxdim=max_dimension,
                       thresh=max_edge_length,
                       distance_matrix=True)
        
        diagrams = result['dgms']
        epsilon = 1e-10
        betti_numbers = []
        
        for dim in range(max_dimension + 1):
            if dim < len(diagrams):
                diagram = diagrams[dim]
                if len(diagram) > 0:
                    births = diagram[:, 0]
                    deaths = diagram[:, 1]
                    persistent_features = np.sum((births <= max_edge_length - epsilon) & 
                                               ((deaths > max_edge_length + epsilon) | 
                                                (deaths == np.inf)))
                    betti_numbers.append(int(persistent_features))
                else:
                    betti_numbers.append(0)
            else:
                betti_numbers.append(0)
        
        return betti_numbers
    
    except Exception as e:
        warnings.warn(f"Error in multi_hom: {str(e)}")
        return [0] * (max_dimension + 1)


def multi_pers(param):
    """
    Computes Betti numbers and persistence diagrams using Ripser for parallel processing.
    
    Args:
        param (tuple): (distance_matrix, max_edge_length)
    
    Returns:
        list: [betti_numbers, persistence_diagram]
    """
    distance_matrix, max_edge_length = param
    max_dimension = 3  # Default
    
    try:
        result = ripser(distance_matrix,
                       maxdim=max_dimension,
                       thresh=max_edge_length,
                       distance_matrix=True)
        
        diagrams = result['dgms']
        
        # Convert to Gudhi-like format for compatibility
        persistence_diagram = []
        for dim, diagram in enumerate(diagrams):
            for birth, death in diagram:
                persistence_diagram.append((dim, birth, death))
        
        # Compute Betti numbers
        epsilon = 1e-10
        betti_numbers = []
        
        for dim in range(max_dimension + 1):
            if dim < len(diagrams):
                diagram = diagrams[dim]
                if len(diagram) > 0:
                    births = diagram[:, 0]
                    deaths = diagram[:, 1]
                    persistent_features = np.sum((births <= max_edge_length - epsilon) & 
                                               ((deaths > max_edge_length + epsilon) | 
                                                (deaths == np.inf)))
                    betti_numbers.append(int(persistent_features))
                else:
                    betti_numbers.append(0)
            else:
                betti_numbers.append(0)
        
        return [betti_numbers, persistence_diagram]
    
    except Exception as e:
        warnings.warn(f"Error in multi_pers: {str(e)}")
        return [[0] * (max_dimension + 1), []]


def wes_dist(param):
    """
    Computes the Wasserstein distance between two persistence diagrams.
    
    This function is compatible with the multiprocessing interface used in
    the original Gudhi implementation.
    
    Args:
        param (tuple): A tuple containing:
            - diagrams (list): A list of two persistence diagrams.
              Each diagram should be either:
              1. A list of (dimension, birth, death) tuples (Gudhi format)
              2. A numpy array of (birth, death) pairs for a specific dimension
            - order (float): The order of the Wasserstein distance (e.g., 1 or 2).
    
    Returns:
        float: The Wasserstein distance between the two diagrams.
    """
    from persim import wasserstein
    
    diagrams, order = param
    
    # Check if diagrams are in Gudhi format (list of (dim, birth, death) tuples)
    # or already in the format expected by persim
    if len(diagrams) != 2:
        raise ValueError("Expected exactly 2 diagrams for comparison")
    
    diag1, diag2 = diagrams[0], diagrams[1]
    
    # If diagrams are lists of tuples with dimension information
    if (isinstance(diag1, list) and len(diag1) > 0 and 
        isinstance(diag1[0], tuple) and len(diag1[0]) == 3):
        # Convert from Gudhi format to persim format
        # Group by dimension
        dims1 = {}
        dims2 = {}
        
        for dim, birth, death in diag1:
            if dim not in dims1:
                dims1[dim] = []
            dims1[dim].append([birth, death])
        
        for dim, birth, death in diag2:
            if dim not in dims2:
                dims2[dim] = []
            dims2[dim].append([birth, death])
        
        # Compute Wasserstein distance for each dimension and sum
        # (This is one common approach; alternatively could compute max)
        total_distance = 0.0
        all_dims = set(dims1.keys()) | set(dims2.keys())
        
        for dim in all_dims:
            points1 = np.array(dims1.get(dim, []))
            points2 = np.array(dims2.get(dim, []))
            
            # Handle empty diagrams
            if len(points1) == 0:
                points1 = np.empty((0, 2))
            if len(points2) == 0:
                points2 = np.empty((0, 2))
            
            # Compute Wasserstein distance for this dimension
            dist = wasserstein(points1, points2, order=order)
            total_distance += dist
        
        return total_distance
    
    else:
        # Assume diagrams are already in the correct format (numpy arrays of (birth, death) pairs)
        # Convert to numpy arrays if needed
        if not isinstance(diag1, np.ndarray):
            diag1 = np.array(diag1)
        if not isinstance(diag2, np.ndarray):
            diag2 = np.array(diag2)
        
        # Ensure 2D arrays
        if diag1.ndim == 1:
            diag1 = diag1.reshape(-1, 2)
        if diag2.ndim == 1:
            diag2 = diag2.reshape(-1, 2)
        
        return wasserstein(diag1, diag2, order=order)


def benchmark_comparison(distance_matrix, max_dimension=2, max_edge_length=1.0):
    """
    Benchmark Ripser against Gudhi for performance comparison.
    
    Args:
        distance_matrix: Input distance matrix
        max_dimension: Maximum homology dimension
        max_edge_length: Maximum filtration value
    
    Returns:
        dict: Timing results and Betti numbers from both implementations
    """
    import time
    
    # Time Ripser
    start_time = time.time()
    betti_ripser = compute_persistent_homology(distance_matrix, 
                                              max_dimension=max_dimension,
                                              max_edge_length=max_edge_length)
    ripser_time = time.time() - start_time
    
    # Try to time Gudhi if available
    gudhi_time = None
    betti_gudhi = None
    try:
        from ..topology.homology import compute_persistent_homology as compute_persistent_homology_gudhi
        
        start_time = time.time()
        betti_gudhi = compute_persistent_homology_gudhi(distance_matrix,
                                                        max_dimension=max_dimension,
                                                        max_edge_length=max_edge_length)
        gudhi_time = time.time() - start_time
    except ImportError:
        print("Gudhi not available for comparison")
    
    results = {
        'ripser_time': ripser_time,
        'ripser_betti': betti_ripser,
        'gudhi_time': gudhi_time,
        'gudhi_betti': betti_gudhi,
        'speedup': gudhi_time / ripser_time if gudhi_time else None
    }
    
    print(f"Ripser time: {ripser_time:.3f}s")
    if gudhi_time:
        print(f"Gudhi time: {gudhi_time:.3f}s")
        print(f"Speedup: {results['speedup']:.1f}x")
    
    return results