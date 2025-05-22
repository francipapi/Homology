import mlx.core as mx
import mlx.nn as nn
import numpy as np
from src.data import dataset
import time as t

def farthest_point_sampling(points, k: int):
    """
    Perform Farthest Point Sampling (FPS) on a set of points using MLX.

    Parameters:
    - points (mlx.ndarray): Input points of shape (N, D), where N is the number of points and D is the dimensionality.
    - k (int): Number of points to sample.

    Returns:
    - sampled_points (mlx.ndarray): Sampled points of shape (k, D).
    """

    mx.set_default_device(mx.gpu)
    points = mx.array(points)
    N, D = points.shape

    # Initialize an array to hold the indices of the sampled points
    sampled_indices = mx.zeros(k, dtype=mx.int32)

    # Initialize a tensor to hold the minimum distances to the sampled points
    distances = mx.full((N,), mx.inf)

    # Randomly select the first point
    sampled_indices[0] = mx.random.randint(0, N)
    last_sampled = points[sampled_indices[0], :]

    for i in range(1, k):
        # Compute squared Euclidean distances from the last sampled point to all points
        diff = points - last_sampled
        dist_sq = mx.sum(diff ** 2, axis=1)

        # Update the minimum distances
        distances = mx.minimum(distances, dist_sq)

        # Select the point with the maximum distance
        sampled_indices[i] = mx.argmax(distances)
        last_sampled = points[sampled_indices[i], :]

    # Gather the sampled points
    sampled_points = points[sampled_indices, :]

    return np.array(sampled_points)
