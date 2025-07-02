"""
Network Simplicial Complex Module

This module provides classes for constructing simplicial complexes from neural network
graphs, particularly focusing on directed flag complexes which are suitable for
directed neural network graphs.

Key Components:
- NetworkSimplicialComplex: Base class for simplicial complex construction
- DirectedFlagComplex: Handles directed graphs from neural networks
- WeightedFiltration: Implements weight-based filtration for persistence
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Union, Any
import graph_tool as gt
from graph_tool import Graph
import subprocess
import tempfile
import os
from abc import ABC, abstractmethod

# Optional imports for different backends
try:
    import gudhi as gd
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    
try:
    import flagser
    FLAGSER_AVAILABLE = True
except ImportError:
    FLAGSER_AVAILABLE = False


class NetworkSimplicialComplex(ABC):
    """
    Abstract base class for constructing simplicial complexes from network graphs.
    """
    
    def __init__(self, max_dimension: int = 2, 
                 max_edge_length: float = 1.0,
                 backend: str = "auto"):
        """
        Initialize simplicial complex builder.
        
        Args:
            max_dimension: Maximum homology dimension to compute
            max_edge_length: Maximum edge weight for filtration
            backend: Backend to use ("gudhi", "flagser", "auto")
        """
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.backend = self._select_backend(backend)
        
    def _select_backend(self, backend: str) -> str:
        """Select appropriate backend based on availability."""
        if backend == "auto":
            if FLAGSER_AVAILABLE:
                return "flagser"
            elif GUDHI_AVAILABLE:
                return "gudhi"
            else:
                raise ImportError("No backend available. Install flagser or gudhi.")
        elif backend == "flagser" and not FLAGSER_AVAILABLE:
            raise ImportError("Flagser not available. Please install it.")
        elif backend == "gudhi" and not GUDHI_AVAILABLE:
            raise ImportError("Gudhi not available. Please install it.")
        return backend
    
    @abstractmethod
    def build_complex(self, graph: Graph) -> Any:
        """Build simplicial complex from graph."""
        pass
    
    @abstractmethod
    def compute_persistence(self, complex: Any) -> Dict[int, np.ndarray]:
        """Compute persistent homology."""
        pass


class DirectedFlagComplex(NetworkSimplicialComplex):
    """
    Constructs directed flag complexes from directed graphs.
    
    A directed flag complex includes all cliques in the directed graph,
    which is suitable for analyzing neural network architectures where
    edge direction matters (information flow).
    """
    
    def build_complex(self, graph: Graph) -> Union[sp.csr_matrix, Any]:
        """
        Build directed flag complex from graph.
        
        Args:
            graph: Directed graph from NetworkGraphBuilder
            
        Returns:
            Complex representation suitable for persistence computation
        """
        # Extract adjacency matrix with weights
        adjacency_matrix = self._graph_to_weighted_adjacency(graph)
        
        if self.backend == "flagser":
            # Flagser works directly with adjacency matrices
            return adjacency_matrix
        elif self.backend == "gudhi":
            # Convert to Gudhi format
            return self._build_gudhi_complex(adjacency_matrix)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _graph_to_weighted_adjacency(self, graph: Graph) -> sp.csr_matrix:
        """Convert graph-tool graph to weighted adjacency matrix."""
        n = graph.num_vertices()
        
        # Get edge list and weights
        edges = []
        weights = []
        
        e_weight = graph.ep.weight
        for e in graph.edges():
            src = int(e.source())
            tgt = int(e.target())
            weight = e_weight[e]
            
            edges.append((src, tgt))
            weights.append(weight)
        
        # Create sparse matrix
        if edges:
            rows, cols = zip(*edges)
            adjacency = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
        else:
            adjacency = sp.csr_matrix((n, n))
        
        return adjacency
    
    def _build_gudhi_complex(self, adjacency: sp.csr_matrix) -> gd.SimplexTree:
        """Build Gudhi simplex tree from adjacency matrix."""
        if not GUDHI_AVAILABLE:
            raise ImportError("Gudhi not available")
            
        st = gd.SimplexTree()
        
        # Add vertices
        for i in range(adjacency.shape[0]):
            st.insert([i], filtration=0.0)
        
        # Add edges with filtration values
        rows, cols = adjacency.nonzero()
        for i, j in zip(rows, cols):
            weight = adjacency[i, j]
            if weight <= self.max_edge_length:
                # Use weight as filtration value
                st.insert([i, j], filtration=weight)
        
        # Expand to get higher-dimensional simplices
        st.expansion(self.max_dimension + 1)
        
        return st
    
    def compute_persistence(self, complex: Any) -> Dict[int, np.ndarray]:
        """
        Compute persistent homology using the selected backend.
        
        Args:
            complex: Complex representation (adjacency matrix or simplex tree)
            
        Returns:
            Dictionary mapping dimension to persistence diagrams
        """
        if self.backend == "flagser":
            return self._compute_flagser_persistence(complex)
        elif self.backend == "gudhi":
            return self._compute_gudhi_persistence(complex)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _compute_flagser_persistence(self, adjacency: sp.csr_matrix) -> Dict[int, np.ndarray]:
        """Compute persistence using flagser."""
        if not FLAGSER_AVAILABLE:
            # Try to use flagser CLI as fallback
            return self._compute_flagser_cli_persistence(adjacency)
        
        # Use python-flagser bindings
        result = flagser.flagser(
            adjacency,
            max_dimension=self.max_dimension,
            directed=True,
            filtration="max"  # Use max of edge weights for simplex filtration
        )
        
        # Extract persistence diagrams
        diagrams = {}
        for dim in range(self.max_dimension + 1):
            if f"dgm{dim}" in result:
                diagrams[dim] = result[f"dgm{dim}"]
            else:
                diagrams[dim] = np.empty((0, 2))
        
        return diagrams
    
    def _compute_flagser_cli_persistence(self, adjacency: sp.csr_matrix) -> Dict[int, np.ndarray]:
        """Compute persistence using flagser command-line tool."""
        # Write adjacency matrix to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.flag', delete=False) as f:
            # Write in flagser format
            f.write(f"dim 0\n")
            for i in range(adjacency.shape[0]):
                f.write(f"{i}\n")
            
            f.write(f"dim 1\n")
            rows, cols = adjacency.nonzero()
            for i, j in zip(rows, cols):
                weight = adjacency[i, j]
                f.write(f"{i} {j} {weight}\n")
            
            temp_file = f.name
        
        try:
            # Run flagser
            result = subprocess.run(
                ['flagser', '--max-dim', str(self.max_dimension), 
                 '--filtration', 'max', temp_file],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Flagser failed: {result.stderr}")
            
            # Parse output
            diagrams = self._parse_flagser_output(result.stdout)
            
        finally:
            # Clean up
            os.unlink(temp_file)
        
        return diagrams
    
    def _parse_flagser_output(self, output: str) -> Dict[int, np.ndarray]:
        """Parse flagser command-line output."""
        diagrams = {}
        current_dim = None
        current_pairs = []
        
        for line in output.split('\n'):
            if line.startswith("persistence intervals in dimension"):
                # Save previous dimension
                if current_dim is not None:
                    diagrams[current_dim] = np.array(current_pairs)
                
                # Start new dimension
                current_dim = int(line.split()[-1].rstrip(':'))
                current_pairs = []
                
            elif line.strip() and current_dim is not None:
                # Parse interval
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        birth = float(parts[0].lstrip('['))
                        death = float(parts[1].rstrip(')'))
                        current_pairs.append([birth, death])
                    except ValueError:
                        pass
        
        # Save last dimension
        if current_dim is not None:
            diagrams[current_dim] = np.array(current_pairs)
        
        # Ensure all dimensions are present
        for dim in range(self.max_dimension + 1):
            if dim not in diagrams:
                diagrams[dim] = np.empty((0, 2))
        
        return diagrams
    
    def _compute_gudhi_persistence(self, simplex_tree: gd.SimplexTree) -> Dict[int, np.ndarray]:
        """Compute persistence using Gudhi."""
        if not GUDHI_AVAILABLE:
            raise ImportError("Gudhi not available")
        
        # Compute persistence
        simplex_tree.compute_persistence()
        
        # Extract persistence diagrams
        persistence = simplex_tree.persistence()
        
        # Organize by dimension
        diagrams = {}
        for dim in range(self.max_dimension + 1):
            pairs = []
            for p in persistence:
                if p[0] == dim:  # Check dimension
                    birth, death = p[1]
                    if death != float('inf') and death - birth > 0:
                        pairs.append([birth, death])
            diagrams[dim] = np.array(pairs) if pairs else np.empty((0, 2))
        
        return diagrams
    
    def compute_betti_numbers(self, complex: Any) -> np.ndarray:
        """
        Compute Betti numbers from the complex.
        
        Args:
            complex: Simplicial complex
            
        Returns:
            Array of Betti numbers for each dimension
        """
        # Compute persistence
        diagrams = self.compute_persistence(complex)
        
        # Extract Betti numbers
        betti_numbers = []
        for dim in range(self.max_dimension + 1):
            if dim in diagrams:
                # Count features with death = infinity or death > max_edge_length
                dgm = diagrams[dim]
                if len(dgm) > 0:
                    persistent_features = np.sum(dgm[:, 1] > self.max_edge_length * 0.99)
                    betti_numbers.append(persistent_features)
                else:
                    betti_numbers.append(0)
            else:
                betti_numbers.append(0)
        
        return np.array(betti_numbers)


class WeightedFiltration(NetworkSimplicialComplex):
    """
    Implements weight-based filtration for neural network graphs.
    
    This class provides methods to create filtrations based on edge weights,
    which correspond to connection strengths in the neural network.
    """
    
    def __init__(self, *args, filtration_type: str = "sublevel", **kwargs):
        """
        Initialize weighted filtration.
        
        Args:
            filtration_type: Type of filtration ("sublevel" or "superlevel")
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.filtration_type = filtration_type
    
    def build_complex(self, graph: Graph) -> Any:
        """Build complex with weight-based filtration."""
        # Delegate to DirectedFlagComplex
        flag_complex = DirectedFlagComplex(
            max_dimension=self.max_dimension,
            max_edge_length=self.max_edge_length,
            backend=self.backend
        )
        return flag_complex.build_complex(graph)
    
    def compute_persistence(self, complex: Any) -> Dict[int, np.ndarray]:
        """Compute persistence with weight-based filtration."""
        # Delegate to DirectedFlagComplex
        flag_complex = DirectedFlagComplex(
            max_dimension=self.max_dimension,
            max_edge_length=self.max_edge_length,
            backend=self.backend
        )
        return flag_complex.compute_persistence(complex)
    
    def create_filtration_values(self, graph: Graph) -> np.ndarray:
        """
        Create filtration values from edge weights.
        
        Args:
            graph: Network graph
            
        Returns:
            Array of filtration values
        """
        # Extract all edge weights
        weights = []
        e_weight = graph.ep.weight
        
        for e in graph.edges():
            weights.append(e_weight[e])
        
        weights = np.array(weights)
        
        if self.filtration_type == "sublevel":
            # Standard filtration: include edges with weight <= threshold
            return weights
        elif self.filtration_type == "superlevel":
            # Reverse filtration: include edges with weight >= threshold
            # Convert by taking max_weight - weight
            if len(weights) > 0:
                max_weight = np.max(weights)
                return max_weight - weights
            else:
                return weights
        else:
            raise ValueError(f"Unknown filtration type: {self.filtration_type}")


def compute_network_homology(graph: Graph, 
                           max_dimension: int = 2,
                           max_edge_length: float = 1.0,
                           backend: str = "auto") -> Dict[str, Any]:
    """
    Convenience function to compute homology of a network graph.
    
    Args:
        graph: Network graph from NetworkGraphBuilder
        max_dimension: Maximum homology dimension
        max_edge_length: Maximum edge weight for filtration
        backend: Backend to use for computation
        
    Returns:
        Dictionary containing:
        - "betti_numbers": Array of Betti numbers
        - "persistence_diagrams": Dict of persistence diagrams by dimension
        - "backend_used": The backend that was used
    """
    # Create directed flag complex
    complex_builder = DirectedFlagComplex(
        max_dimension=max_dimension,
        max_edge_length=max_edge_length,
        backend=backend
    )
    
    # Build complex
    complex = complex_builder.build_complex(graph)
    
    # Compute persistence
    diagrams = complex_builder.compute_persistence(complex)
    
    # Compute Betti numbers
    betti_numbers = complex_builder.compute_betti_numbers(complex)
    
    return {
        "betti_numbers": betti_numbers,
        "persistence_diagrams": diagrams,
        "backend_used": complex_builder.backend
    }