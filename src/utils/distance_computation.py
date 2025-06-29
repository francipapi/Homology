import torch
import numpy as np
import yaml
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_distances
import scipy as sp
import graph_tool as gt
from graph_tool.topology import shortest_distance
from typing import Tuple, Optional, Union
import time


def load_config(config_path: str = "configs/homology_config.yaml") -> dict:
    """Load homology configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_cosine_distance_matrix(X: Union[np.ndarray, torch.Tensor], device: str = 'auto') -> np.ndarray:
    """
    Compute cosine distance matrix for a set of points.
    
    Cosine distance = 1 - cosine_similarity
    where cosine_similarity(u, v) = dot(u, v) / (norm(u) * norm(v))
    
    Parameters:
    - X: Input points of shape (N, D), where N is the number of points and D is the dimensionality
    - device: Device to use ('auto', 'cpu', 'cuda', 'mps') for PyTorch computation
    
    Returns:
    - distance_matrix: Cosine distance matrix of shape (N, N)
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Convert to torch tensor if needed
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    elif isinstance(X, torch.Tensor):
        X_tensor = X.float().to(device)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    
    # Normalize each vector to unit length
    X_normalized = X_tensor / (torch.norm(X_tensor, dim=1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity matrix
    cosine_sim = torch.mm(X_normalized, X_normalized.t())
    
    # Convert to cosine distance (1 - cosine_similarity)
    cosine_dist = 1.0 - cosine_sim
    
    # Ensure diagonal is exactly zero
    cosine_dist.fill_diagonal_(0.0)
    
    # Ensure non-negative distances (handle numerical errors)
    cosine_dist = torch.clamp(cosine_dist, min=0.0)
    
    # Convert back to numpy
    if cosine_dist.is_cuda or hasattr(cosine_dist, 'cpu'):
        return cosine_dist.cpu().numpy()
    else:
        return cosine_dist.numpy()


def farthest_point_sampling_pytorch(points: Union[np.ndarray, torch.Tensor], device: str = 'auto', k: Optional[int] = None, metric: Optional[str] = None) -> np.ndarray:
    """
    Perform Farthest Point Sampling (FPS) on a set of points using PyTorch.
    
    Parameters:
    - points: Input points of shape (N, D), where N is the number of points and D is the dimensionality
    - device: Device to use ('auto', 'cpu', 'cuda', 'mps')
    - k: Number of points to sample (uses config if None)
    - metric: Distance metric to use ('euclidean' or 'cosine') (uses config if None)
    
    Returns:
    - sampled_points: Sampled points of shape (k, D) as numpy array
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    config = load_config()
    if k is None:
        k = config['sampling']['fps_num_points']
    if metric is None:
        metric = config['distance'].get('metric', 'euclidean')
        # Handle typo in config
        if metric == 'euclidian':
            metric = 'euclidean'
    normalization = config['sampling'].get('normalization', False)
    
    # Convert to torch tensor if needed
    if isinstance(points, np.ndarray):
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    elif isinstance(points, torch.Tensor):
        points_tensor = points.float().to(device)
    else:
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    
    # Apply normalization if enabled
    if normalization:
        # Normalize along each dimension (feature-wise normalization)
        points_tensor = (points_tensor - points_tensor.mean(dim=0, keepdim=True)) / (points_tensor.std(dim=0, keepdim=True) + 1e-8)
    
    N, D = points_tensor.shape
    
    if k >= N:
        if isinstance(points, np.ndarray):
            return points
        elif isinstance(points, torch.Tensor):
            return points.cpu().numpy()
        else:
            return np.array(points)
    
    # Initialize arrays
    sampled_indices = torch.zeros(k, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    
    # Randomly select first point
    sampled_indices[0] = torch.randint(0, N, (1,), device=device)
    last_sampled = points_tensor[sampled_indices[0], :]
    
    for i in range(1, k):
        if metric == 'cosine':
            # Compute cosine distances from last sampled point to all points
            # Normalize the last sampled point
            last_norm = last_sampled / (torch.norm(last_sampled) + 1e-8)
            # Normalize all points
            points_norm = points_tensor / (torch.norm(points_tensor, dim=1, keepdim=True) + 1e-8)
            # Compute cosine similarity and convert to distance
            cosine_sim = torch.mv(points_norm, last_norm)
            dist = 1.0 - cosine_sim
        else:  # euclidean
            # Compute squared Euclidean distances from last sampled point to all points
            diff = points_tensor - last_sampled.unsqueeze(0)
            dist = torch.sum(diff ** 2, dim=1)
        
        # Update minimum distances
        distances = torch.minimum(distances, dist)
        
        # Select point with maximum distance
        sampled_indices[i] = torch.argmax(distances)
        last_sampled = points_tensor[sampled_indices[i], :]
    
    # Return sampled points as numpy array
    sampled_points = points_tensor[sampled_indices, :]
    if sampled_points.is_cuda or hasattr(sampled_points, 'cpu'):
        return sampled_points.cpu().numpy()
    else:
        return sampled_points.numpy()


def knn_geodesic_distance(X: np.ndarray, k: Optional[int] = None, use_fps: Optional[bool] = None, metric: Optional[str] = None) -> np.ndarray:
    """
    Compute geodesic distance matrix using k-nearest neighbors graph.
    Ported from original graph.py distance() function using graph_tool.
    
    Parameters:
    - X: Input points of shape (N, D) - should already be normalized if required
    - k: Number of nearest neighbors (uses config if None)
    - use_fps: Whether to use furthest point sampling (uses config if None)
    - metric: Distance metric to use ('euclidean', 'cosine', 'manhattan', etc.) (uses config if None)
    
    Returns:
    - distance_matrix: Integer geodesic distance matrix of shape (N, N)
    """
    config = load_config()
    if k is None:
        k = config['distance']['k_neighbors']
    if use_fps is None:
        use_fps = config['sampling']['use_fps']
    if metric is None:
        metric = config['distance'].get('metric', 'euclidean')
        # Handle typo in config
        if metric == 'euclidian':
            metric = 'euclidean'

    if use_fps:
        X = farthest_point_sampling_pytorch(X)

    # sklearn's kneighbors_graph supports various metrics including cosine
    graph = kneighbors_graph(X, k, mode='connectivity', metric=metric, n_jobs=-1)
    g = gt.Graph(sp.sparse.lil_matrix(graph), directed=False)
    distance_matrix = shortest_distance(g)
    # Convert to integer array as geodesic distances are edge counts
    return np.array(distance_matrix.get_2d_array(), dtype=np.int32)
