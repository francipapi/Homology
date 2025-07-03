"""
Network Geodesic Distance Module

This module provides efficient computation of geodesic (shortest path) distances
on neural network graphs, following the approach from nn-evolution.

The geodesic distance captures the effective distance between neurons considering
all possible paths through the network, which is more informative than direct
edge weights for topological analysis.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from typing import Optional, Union, Tuple
import graph_tool as gt
from graph_tool.topology import shortest_distance
import warnings


class GraphGeodesicDistance:
    """
    Computes geodesic distances on graphs for persistent homology computation.
    
    This class provides multiple backends for computing all-pairs shortest paths
    on weighted directed graphs, optimized for neural network architectures.
    """
    
    def __init__(self, backend: str = "auto", directed: bool = True):
        """
        Initialize geodesic distance computer.
        
        Args:
            backend: Backend to use ("graph-tool", "scipy", "auto")
            directed: Whether to treat graph as directed
        """
        self.backend = self._select_backend(backend)
        self.directed = directed
        
    def _select_backend(self, backend: str) -> str:
        """Select appropriate backend based on availability."""
        if backend == "auto":
            # Prefer graph-tool for performance
            try:
                import graph_tool
                return "graph-tool"
            except ImportError:
                return "scipy"
        return backend
    
    def compute_distances(self, adjacency: Union[sp.spmatrix, gt.Graph]) -> sp.csr_matrix:
        """
        Compute geodesic distances from adjacency representation.
        
        Args:
            adjacency: Sparse adjacency matrix or graph-tool Graph
            
        Returns:
            Sparse matrix of geodesic distances
        """
        if isinstance(adjacency, gt.Graph):
            return self._compute_graph_tool_distances(adjacency)
        else:
            # Assume sparse matrix
            return self._compute_scipy_distances(adjacency)
    
    def _compute_scipy_distances(self, adjacency: sp.spmatrix) -> sp.csr_matrix:
        """
        Compute distances using scipy's shortest_path.
        
        Args:
            adjacency: Sparse adjacency matrix with edge weights
            
        Returns:
            Matrix of shortest path distances
        """
        # Ensure CSR format for efficiency
        if not isinstance(adjacency, sp.csr_matrix):
            adjacency = adjacency.tocsr()
        
        # Compute shortest paths
        # Note: scipy expects weights as "distances" already
        # So if using nn-evolution normalization, weights should be distances
        distances = shortest_path(
            adjacency,
            method='D',  # Dijkstra's algorithm
            directed=self.directed,
            return_predecessors=False
        )
        
        # Handle disconnected components (infinite distances)
        # Replace inf with a large value (following nn-evolution)
        max_finite = np.max(distances[np.isfinite(distances)]) if np.any(np.isfinite(distances)) else 1.0
        distances[np.isinf(distances)] = max_finite + 1.0
        
        # Convert to sparse format to save memory
        # Only keep distances below a threshold
        threshold = max_finite + 0.5
        distances[distances > threshold] = 0
        
        return sp.csr_matrix(distances)
    
    def _compute_graph_tool_distances(self, graph: gt.Graph) -> sp.csr_matrix:
        """
        Compute distances using graph-tool (more efficient for large graphs).
        
        Args:
            graph: graph-tool Graph object
            
        Returns:
            Matrix of shortest path distances
        """
        n = graph.num_vertices()
        
        # Get edge weights
        e_weight = graph.ep.weight
        
        # Create distance matrix
        # For memory efficiency, we'll compute in batches
        batch_size = min(100, n)
        distances = sp.lil_matrix((n, n))
        
        for start_batch in range(0, n, batch_size):
            end_batch = min(start_batch + batch_size, n)
            
            # Compute distances from this batch of vertices
            for v_idx in range(start_batch, end_batch):
                v = graph.vertex(v_idx)
                
                # Compute shortest distances from v to all other vertices
                dist_map = shortest_distance(
                    graph, 
                    source=v,
                    weights=e_weight,
                    directed=self.directed
                )
                
                # Store in matrix
                for target_idx in range(n):
                    d = dist_map[graph.vertex(target_idx)]
                    if d < float('inf'):
                        distances[v_idx, target_idx] = d
        
        # Convert to CSR format
        distances = distances.tocsr()
        
        # Handle disconnected components
        # Find maximum finite distance
        if distances.nnz > 0:
            max_finite = distances.data.max()
        else:
            max_finite = 1.0
            
        # For efficiency, we don't explicitly set infinite distances
        # They remain as implicit zeros in the sparse matrix
        
        return distances
    
    def compute_distances_from_weights(self, weights: np.ndarray, 
                                     shape: Tuple[int, int],
                                     edge_list: np.ndarray) -> sp.csr_matrix:
        """
        Compute geodesic distances from edge weights and connectivity.
        
        This is useful when you have the network weights but haven't built
        the full graph structure yet.
        
        Args:
            weights: Array of edge weights (already normalized as distances)
            shape: Shape of adjacency matrix (n_vertices, n_vertices)
            edge_list: Array of shape (n_edges, 2) with source/target pairs
            
        Returns:
            Sparse matrix of geodesic distances
        """
        # Create sparse adjacency matrix
        rows = edge_list[:, 0]
        cols = edge_list[:, 1]
        
        adjacency = sp.csr_matrix(
            (weights, (rows, cols)),
            shape=shape
        )
        
        return self._compute_scipy_distances(adjacency)
    
    def filter_distances(self, distances: sp.csr_matrix, 
                        epsilon: float = 0.6) -> sp.csr_matrix:
        """
        Apply epsilon filtering as in nn-evolution.
        
        This removes edges with distance > epsilon, which helps reduce noise
        in the persistent homology computation.
        
        Args:
            distances: Geodesic distance matrix
            epsilon: Maximum distance to keep
            
        Returns:
            Filtered distance matrix
        """
        # Create a copy to avoid modifying the original
        filtered = distances.copy()
        
        # Remove distances greater than epsilon
        filtered.data[filtered.data > epsilon] = 0
        filtered.eliminate_zeros()
        
        return filtered
    
    def validate_distance_matrix(self, distances: sp.csr_matrix) -> bool:
        """
        Validate that the distance matrix has expected properties.
        
        Args:
            distances: Distance matrix to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check for negative distances
        if distances.data.min() < 0:
            raise ValueError("Distance matrix contains negative values")
        
        # Check diagonal (should be zero for connected vertices)
        diagonal = distances.diagonal()
        if not np.allclose(diagonal[diagonal != 0], 0, atol=1e-10):
            warnings.warn("Distance matrix has non-zero diagonal elements")
        
        # Check for NaN or inf
        if np.any(np.isnan(distances.data)):
            raise ValueError("Distance matrix contains NaN values")
        
        if np.any(np.isinf(distances.data)):
            warnings.warn("Distance matrix contains infinite values")
        
        return True


def compute_network_geodesic_distances(graph: gt.Graph,
                                     epsilon: Optional[float] = 0.6,
                                     backend: str = "auto") -> sp.csr_matrix:
    """
    Convenience function to compute geodesic distances for a neural network graph.
    
    This follows the nn-evolution approach:
    1. Compute all-pairs shortest paths
    2. Apply epsilon filtering
    3. Return sparse distance matrix
    
    Args:
        graph: Network graph (with weights as distances)
        epsilon: Maximum distance for filtering (None to skip)
        backend: Backend to use for computation
        
    Returns:
        Sparse matrix of filtered geodesic distances
    """
    # Initialize distance computer
    distance_computer = GraphGeodesicDistance(backend=backend, directed=True)
    
    # Compute distances
    distances = distance_computer.compute_distances(graph)
    
    # Apply filtering if requested
    if epsilon is not None:
        distances = distance_computer.filter_distances(distances, epsilon)
    
    # Validate
    distance_computer.validate_distance_matrix(distances)
    
    return distances