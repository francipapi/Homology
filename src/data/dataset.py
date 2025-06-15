import trimesh as tr 
import numpy as np
import torch
import plotly.graph_objects as go

def generate(n, big_radius, small_radius):
    # Helper function for creating and transforming a torus pair
    def create_transformed_torus_pair(offset, rotation_axis, rotation_angle, translation_vector):
        torus1 = tr.creation.torus(big_radius, small_radius)
        torus2 = tr.creation.torus(big_radius, small_radius)
        
        # Apply rotation to the second torus
        rotation_matrix = tr.transformations.rotation_matrix(rotation_angle, rotation_axis)
        torus2.apply_transform(rotation_matrix)
        
        # Apply translation to separate the tori
        translation_matrix1 = tr.transformations.translation_matrix([big_radius/2, 0, 0])
        translation_matrix2 = tr.transformations.translation_matrix([-big_radius/2, 0, 0])
        torus2.apply_transform(translation_matrix2)
        torus1.apply_transform(translation_matrix1)
        
        # Apply offsets for positioning
        torus1.apply_transform(tr.transformations.translation_matrix(translation_vector))
        torus2.apply_transform(tr.transformations.translation_matrix(translation_vector))
        
        return torus1, torus2

    # Define translations
    scale_factor = big_radius * 3
    translations = [
        [-scale_factor, scale_factor, scale_factor],
        [-scale_factor, -scale_factor, scale_factor],
        [-scale_factor, scale_factor, -scale_factor],
        [-scale_factor, -scale_factor, -scale_factor],
        [scale_factor, scale_factor, scale_factor],
        [scale_factor, -scale_factor, scale_factor],
        [scale_factor, scale_factor, -scale_factor],
        [scale_factor, -scale_factor, -scale_factor]
    ]
    
    # Create tori pairs with transformations
    torus_pairs = []
    for translation in translations:
        torus1, torus2 = create_transformed_torus_pair(
            offset=big_radius, 
            rotation_axis=[1, 0, 0], 
            rotation_angle=np.pi / 2, 
            translation_vector=translation
        )
        torus_pairs.extend([torus1, torus2])
    
    # Sample points from all the tori
    sampled_points = []
    labels = []
    for i, torus in enumerate(torus_pairs):
        points = np.array(torus.sample(n))
        sampled_points.append(points)
        labels.append(np.full((n, 1), i % 2))  # Alternating labels 0 and 1
    
    # Concatenate results
    X = np.concatenate(sampled_points)
    y = np.concatenate(labels)
    
    return [X, y]

def gen_easy(n, big_radius, small_radious):
    torus1 = tr.creation.torus(big_radius,small_radious)
    torus2 = tr.creation.torus(big_radius,small_radious)
    rotation_matrix = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    torus2.apply_transform(rotation_matrix)
    translation_matrix1 = tr.transformations.translation_matrix([big_radius/2, 0, 0])
    translation_matrix2 = tr.transformations.translation_matrix([-big_radius/2, 0, 0])
    torus2.apply_transform(translation_matrix2)
    torus1.apply_transform(translation_matrix1)
    points1 = np.array(torus1.sample(n))
    points2 = np.array(torus2.sample(n))
    X = np.concatenate((points1, points2))
    y = np.concatenate((np.zeros((n,1)), np.ones((n,1))))
    return [X,y]

def plot_torus_points(X, y, filename=None):
    """
    Plots the generated torus points using Plotly.

    Parameters:
    - X (numpy.ndarray): Array of shape (N, 3) representing the 3D coordinates.
    - y (numpy.ndarray): Labels corresponding to each point in X.
    - filename (str or Path, optional): If provided, saves the plot to this file instead of displaying it.
    """
    # Extract x, y, z coordinates
    x_coords = X[:, 0]
    y_coords = X[:, 1]
    z_coords = X[:, 2]

    # Create a scatter plot with Plotly
    fig = go.Figure()

    # Add points to the figure, coloring them based on their labels
    for label in np.unique(y):
        mask = y.flatten() == label
        fig.add_trace(
            go.Scatter3d(
                x=x_coords[mask],
                y=y_coords[mask],
                z=z_coords[mask],
                mode='markers',
                marker=dict(size=2),
                name=f"Label {int(label)}"
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title="3D Visualization of Torus Points",
        scene=dict(
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            zaxis_title="Z-axis"
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Show the figure or save it to a file
    if filename:
        fig.write_image(str(filename))
    else:
        fig.show()

def farthest_point_sampling(point_cloud, num_samples):
    """
    A simple implementation of farthest point sampling.
    
    Parameters:
    - point_cloud: np.ndarray of shape (N, D), where N is the number of points and D is the dimension.
    - num_samples: int, number of points to sample.
    
    Returns:
    - sampled_points: np.ndarray of shape (num_samples, D), sampled point cloud.
    """
    if not isinstance(point_cloud, np.ndarray):
        point_cloud = np.array(point_cloud)
    
    N = point_cloud.shape[0]
    if num_samples >= N:
        return point_cloud
    
    # Initialize the sampled points array
    sampled_points = np.zeros((num_samples, point_cloud.shape[1]))
    
    # Randomly select the first point
    sampled_points[0] = point_cloud[np.random.randint(N)]
    
    # Compute distances to the first point
    distances = np.linalg.norm(point_cloud - sampled_points[0], axis=1)
    
    # Iteratively select points
    for i in range(1, num_samples):
        # Find the point farthest from all currently sampled points
        farthest_idx = np.argmax(distances)
        sampled_points[i] = point_cloud[farthest_idx]
        
        # Update distances
        new_distances = np.linalg.norm(point_cloud - sampled_points[i], axis=1)
        distances = np.minimum(distances, new_distances)
    
    return sampled_points

def parallel_farthest_point_sampling(param):
    point_cloud, num_samples = param
    return farthest_point_sampling(point_cloud, num_samples)

def parallel_bucket_point_sampling(param):
    point_cloud, num_samples = param
    return farthest_point_sampling(point_cloud, num_samples)  # Using FPS as fallback

''' TEST

# Example usage:
X, y = gen_easy(4000,3,1)
original_pc = torch.tensor(X, dtype=torch.float)  # Replace with your point cloud
plot_torus_points(X,y)
num_samples = 2000  # Desired number of points
uniform_pc = farthest_point_sampling(original_pc, num_samples)
plot_torus_points(uniform_pc, np.zeros(uniform_pc.shape[0]))

'''
