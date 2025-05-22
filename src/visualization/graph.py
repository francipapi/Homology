from sklearn.neighbors import kneighbors_graph
import numpy as np 
import time as t 
import graph_tool as gt
from graph_tool.topology import shortest_distance
import scipy as sp
from persim import plot_diagrams

def distance(X,k):
    graph = kneighbors_graph(X, k, mode='connectivity', p=2, n_jobs=-1)
    g = gt.Graph(sp.sparse.lil_matrix(graph), directed=False)
    distance_matrix = shortest_distance(g)
    return np.array(distance_matrix.get_2d_array())

def multi_distance(param):
    X, k = param 
    graph = kneighbors_graph(X, k, mode='connectivity', n_jobs=-1)
    g = gt.Graph(sp.sparse.lil_matrix(graph), directed=False)
    distance_matrix = shortest_distance(g)
    return np.array(distance_matrix.get_2d_array())

def compute_distance_matrix(X,k):
    """
    Computes the distance matrix for a given graph-tool graph.

    Parameters:
    - g (graph_tool.Graph): The input graph.

    Returns:
    - distance_matrix (numpy.ndarray): A 2D NumPy array representing the distance matrix.
    """
    graph = kneighbors_graph(X, k, mode='connectivity', n_jobs=-1)
    g = gt.Graph(sp.sparse.lil_matrix(graph), directed=False)
    num_vertices = g.num_vertices()
    distance_matrix = np.full((num_vertices, num_vertices), np.inf)  # Initialize with infinity

    # Iterate over each vertex and compute shortest paths
    for v in g.vertices():
        distances = shortest_distance(g, source=v)
        source_index = int(v)
        for u, distance in enumerate(distances):
            distance_matrix[source_index, u] = distance

    return distance_matrix

