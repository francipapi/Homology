import torch
import numpy as np
import yaml
from sklearn.neighbors import kneighbors_graph
import scipy as sp
import graph_tool as gt
from graph_tool.topology import shortest_distance
from typing import Tuple, Optional, Union
import time


def load_config(config_path: str = "configs/homology_config.yaml") -> dict:
    """Load homology configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def farthest_point_sampling_pytorch(points: Union[np.ndarray, torch.Tensor], device: str = 'auto') -> np.ndarray:
    """
    Perform Farthest Point Sampling (FPS) on a set of points using PyTorch.
    
    Parameters:
    - points: Input points of shape (N, D), where N is the number of points and D is the dimensionality
    - device: Device to use ('auto', 'cpu', 'cuda', 'mps')
    
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
    k = config['sampling']['fps_num_points']
    # Convert to torch tensor if needed
    if isinstance(points, np.ndarray):
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    elif isinstance(points, torch.Tensor):
        points_tensor = points.float().to(device)
    else:
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    
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
        # Compute squared Euclidean distances from last sampled point to all points
        diff = points_tensor - last_sampled.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=1)
        
        # Update minimum distances
        distances = torch.minimum(distances, dist_sq)
        
        # Select point with maximum distance
        sampled_indices[i] = torch.argmax(distances)
        last_sampled = points_tensor[sampled_indices[i], :]
    
    # Return sampled points as numpy array
    sampled_points = points_tensor[sampled_indices, :]
    if sampled_points.is_cuda or hasattr(sampled_points, 'cpu'):
        return sampled_points.cpu().numpy()
    else:
        return sampled_points.numpy()


def knn_geodesic_distance(X: np.ndarray) -> np.ndarray:
    """
    Compute geodesic distance matrix using k-nearest neighbors graph.
    Ported from original graph.py distance() function using graph_tool.
    
    Parameters:
    - X: Input points of shape (N, D)
    
    Returns:
    - distance_matrix: Integer geodesic distance matrix of shape (N, N)
    """
    config = load_config()
    k = config['distance']['k_neighbors']

    if config['sampling']['use_fps']:
        X = farthest_point_sampling_pytorch(X)

    graph = kneighbors_graph(X, k, mode='connectivity', p=2, n_jobs=-1)
    g = gt.Graph(sp.sparse.lil_matrix(graph), directed=False)
    distance_matrix = shortest_distance(g)
    # Convert to integer array as geodesic distances are edge counts
    return np.array(distance_matrix.get_2d_array(), dtype=np.int32)
