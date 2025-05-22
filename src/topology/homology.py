import gudhi as gd
import matplotlib.pyplot as plt
import os
import numpy as np
from gudhi.wasserstein import wasserstein_distance

def compute_persistent_homology(distance_matrix, max_dimension=2, max_edge_length=1,
                                diag_filename='persistence_diagram.txt',
                                plot_barcode_filename='persistence_diagram.png',
                                betti_filename='betti_numbers.txt'):
    """
    Computes the persistent homology of a given distance matrix using Gudhi,
    and saves the persistence diagram and Betti numbers.

    Parameters:
    - distance_matrix (numpy.ndarray): A 2D NumPy array representing the distance matrix.
    - max_dimension (int): The maximum homology dimension to compute. Default is 2.
    - max_edge_length (float): The maximum edge length for the Vietoris-Rips complex. Default is infinity.
    - diag_filename (str): Filename to save the persistence diagram as a text file.
    - plot_filename (str): Filename to save the persistence diagram plot as an image.
    - betti_filename (str): Filename to save the Betti numbers as a text file.

    Returns:
    - None
    """
    
    # Validate the distance matrix
    if not isinstance(distance_matrix, np.ndarray):
        raise ValueError("distance_matrix must be a NumPy array.")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square (n x n) matrix.")
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("distance_matrix must be symmetric.")
    if not np.all(np.diag(distance_matrix) == 0):
        raise ValueError("The diagonal of distance_matrix must be all zeros.")
    
    # Number of points
    num_points = distance_matrix.shape[0]

    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    simplex_tree.collapse_edges()
    simplex_tree.expansion(max_dimension)
    persistence = simplex_tree.persistence()

    return simplex_tree.betti_numbers()

def multi_hom(param):
    distance_matrix, max_edge_length = param
    max_dimension=3

    # Validate the distance matrix
    if not isinstance(distance_matrix, np.ndarray):
        raise ValueError("distance_matrix must be a NumPy array.")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square (n x n) matrix.")
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("distance_matrix must be symmetric.")
    if not np.all(np.diag(distance_matrix) == 0):
        raise ValueError("The diagonal of distance_matrix must be all zeros.")
    
    # Number of points
    num_points = distance_matrix.shape[0]

    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    simplex_tree.collapse_edges()
    simplex_tree.expansion(max_dimension)
    persistence = simplex_tree.persistence()

    return simplex_tree.betti_numbers()

def multi_pers(param):
    distance_matrix, max_edge_length = param
    max_dimension=3

    # Validate the distance matrix
    if not isinstance(distance_matrix, np.ndarray):
        raise ValueError("distance_matrix must be a NumPy array.")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square (n x n) matrix.")
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("distance_matrix must be symmetric.")
    if not np.all(np.diag(distance_matrix) == 0):
        raise ValueError("The diagonal of distance_matrix must be all zeros.")
    
    # Number of points
    num_points = distance_matrix.shape[0]

    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    simplex_tree.collapse_edges()
    simplex_tree.expansion(max_dimension)
    persistence = simplex_tree.persistence(min_persistence=max_edge_length-1)

    return [simplex_tree.betti_numbers(), persistence]

def wes_dist(param):
    diagrams, order = param
    return wasserstein_distance(diagrams[0], diagrams[1], order=order)

