"""
Computes topological features of data, primarily focusing on persistent homology.

This module uses the Gudhi library to construct Vietoris-Rips complexes from
distance matrices and then computes persistent homology. It provides functions
to calculate Betti numbers, persistence diagrams, and barcode plots.
Additionally, it includes helper functions for batch processing and computing
Wasserstein distances between persistence diagrams, which are useful for
comparing topological similarity.
"""
import gudhi as gd
import matplotlib.pyplot as plt
import os # Used for path operations, though not explicitly in current functions
import numpy as np
from gudhi.wasserstein import wasserstein_distance

def compute_persistent_homology(distance_matrix, max_dimension=2, max_edge_length=1.0,
                                diag_filename=None,
                                plot_barcode_filename=None,
                                betti_filename=None):
    """
    Computes persistent homology of a dataset represented by a distance matrix.

    Constructs a Vietoris-Rips complex and calculates its persistent homology
    up to `max_dimension`. Optionally, saves the persistence diagram (birth-death pairs),
    a barcode plot visualization, and the Betti numbers to specified files.

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
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-6): # Added tolerance for symmetry check
        raise ValueError("distance_matrix must be symmetric.")
    if not np.all(np.diag(distance_matrix) < 1e-6): # Check if diagonal is close to zero
        raise ValueError("The diagonal of distance_matrix must be all zeros.")
    
    # Construct Vietoris-Rips complex from the distance matrix.
    # max_edge_length defines the maximum scale for the filtration.
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    
    # Create a simplex tree from the Rips complex.
    # max_dimension=1 initially, then expanded. This is a common Gudhi pattern.
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    
    # Optional: Collapse edges to simplify the complex. May not always be necessary or desired.
    simplex_tree.collapse_edges() 
    
    # Expand the simplex tree to the specified maximum dimension for homology computation.
    simplex_tree.expansion(max_dimension)
    
    # Compute persistence pairs (birth, death) for topological features.
    persistence = simplex_tree.persistence()

    # Save persistence diagram to a file if a filename is provided.
    if diag_filename:
        # Ensure the directory exists before writing
        os.makedirs(os.path.dirname(diag_filename), exist_ok=True)
        with open(diag_filename, 'w') as f:
            for dim, (birth, death) in persistence:
                f.write(f"{dim} {birth} {death}\n")
        print(f"Persistence diagram saved to {diag_filename}")

    # Generate and save a barcode plot if a filename is provided.
    if plot_barcode_filename:
        os.makedirs(os.path.dirname(plot_barcode_filename), exist_ok=True)
        gd.plot_persistence_barcode(persistence)
        plt.title(f"Persistence Barcode (max_edge_length={max_edge_length})") # Add a title
        plt.savefig(plot_barcode_filename)
        plt.close()  # Close the plot to free memory and prevent interactive display.
        print(f"Barcode plot saved to {plot_barcode_filename}")

    # Compute Betti numbers (number of holes in each dimension).
    betti_numbers = simplex_tree.betti_numbers()
    # Gudhi might return more Betti numbers than max_dimension if simplex_tree was expanded further.
    # Truncate to max_dimension + 1 (for H0...H_max_dimension).
    if len(betti_numbers) > max_dimension + 1:
        betti_numbers = betti_numbers[:max_dimension + 1]


    # Save Betti numbers to a file if a filename is provided.
    if betti_filename:
        os.makedirs(os.path.dirname(betti_filename), exist_ok=True)
        np.savetxt(betti_filename, np.array(betti_numbers), fmt='%d', header=f"Betti numbers for max_dimension={max_dimension}")
        print(f"Betti numbers saved to {betti_filename}")

    return betti_numbers

# The following functions (multi_hom, multi_pers, wes_dist) seem to be designed
# for specific batch processing or comparative analyses, possibly with a parallel execution framework
# (implied by 'param' argument structure common in multiprocessing).
# Their docstrings are updated for clarity assuming this context.

def multi_hom(param):
    """
    Computes Betti numbers for a single dataset in a batched or parallel context.

    This function is likely a target for `multiprocessing.Pool.map` or similar,
    where `param` is a tuple containing the necessary arguments.

    Args:
        param (tuple): A tuple containing:
            - distance_matrix (np.ndarray): The distance matrix of the dataset.
            - max_edge_length (float): Maximum edge length for Rips complex.
            (Implicitly, max_dimension might be fixed or globally set for such batch operations).

    Returns:
        list[int]: Betti numbers for the dataset.
    """
    distance_matrix, max_edge_length = param
    # Defaulting max_dimension for this specific batch function. Could be passed in param.
    max_dimension = 3 

    # (Input validation for distance_matrix is repeated here; could be refactored)
    if not isinstance(distance_matrix, np.ndarray):
        raise ValueError("distance_matrix must be a NumPy array.")
    # ... (other validations as in compute_persistent_homology) ...
    
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1) # Initial creation
    simplex_tree.collapse_edges()
    simplex_tree.expansion(max_dimension) # Expand to desired dimension
    # No persistence pairs computed explicitly if only Betti numbers are needed for this function.
    
    betti_numbers = simplex_tree.betti_numbers()
    if len(betti_numbers) > max_dimension + 1:
        betti_numbers = betti_numbers[:max_dimension + 1]
    return betti_numbers

def multi_pers(param):
    """
    Computes Betti numbers and persistence diagram for a single dataset in a batched context.

    Args:
        param (tuple): A tuple containing:
            - distance_matrix (np.ndarray): The distance matrix.
            - max_edge_length (float): Maximum edge length for Rips complex.
            (Implicitly, max_dimension might be fixed).

    Returns:
        list: A list containing two elements:
            - list[int]: Betti numbers.
            - list[tuple]: Persistence diagram (dim, birth, death).
    """
    distance_matrix, max_edge_length = param
    max_dimension = 3 # Defaulting max_dimension

    # (Input validation for distance_matrix)
    if not isinstance(distance_matrix, np.ndarray):
        raise ValueError("distance_matrix must be a NumPy array.")
    # ... (other validations) ...

    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    simplex_tree.collapse_edges()
    simplex_tree.expansion(max_dimension)
    
    # min_persistence filters out short-lived features. Here, it's set based on max_edge_length,
    # which might be an unusual choice (usually a small absolute value).
    # Consider making min_persistence a separate parameter if this function is actively used.
    persistence_diagram = simplex_tree.persistence(min_persistence=max_edge_length - 1) 

    betti_numbers = simplex_tree.betti_numbers()
    if len(betti_numbers) > max_dimension + 1:
        betti_numbers = betti_numbers[:max_dimension + 1]
        
    return [betti_numbers, persistence_diagram]

def wes_dist(param): # 'wes_dist' likely means Wasserstein distance
    """
    Computes the Wasserstein distance between two persistence diagrams.

    Args:
        param (tuple): A tuple containing:
            - diagrams (list[list[tuple]]): A list of two persistence diagrams.
              Each diagram is a list of (dimension, birth, death) tuples.
            - order (float): The order of the Wasserstein distance (e.g., 1 or 2).

    Returns:
        float: The Wasserstein distance between the two diagrams.
    """
    diagrams, order = param
    # Ensure diagrams are in the format Gudhi expects if they are raw persistence pairs
    # Gudhi's wasserstein_distance expects lists of (birth, death) points for a fixed dimension.
    # This function might need preprocessing if `diagrams` are lists of (dim, birth, death).
    # Assuming `diagrams[0]` and `diagrams[1]` are already filtered for a specific dimension
    # and are lists of (birth, death) pairs. If not, this will error or give incorrect results.
    # For robust use, this function should handle filtering by dimension.
    
    # Example for handling:
    # diag1_dim_X = np.array([p[1:] for p in diagrams[0] if p[0] == X]) # Filter for dimension X
    # diag2_dim_X = np.array([p[1:] for p in diagrams[1] if p[0] == X]) # Filter for dimension X
    # return wasserstein_distance(diag1_dim_X, diag2_dim_X, order=order)
    # For now, assuming inputs are correctly formatted for direct use:
    return wasserstein_distance(diagrams[0], diagrams[1], order=order)
