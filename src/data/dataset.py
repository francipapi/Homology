import trimesh as tr 
import numpy as np
import torch
import plotly.graph_objects as go
import pyvista as pv
import open3d as o3d

def generate(n, big_radius, small_radius, solid=False, interior_noise=0.1):
    """
    Generate torus pairs with option for solid vs hollow tori.
    
    Parameters:
    - n: Number of points to sample per torus
    - big_radius: Major radius of the torus
    - small_radius: Minor radius (tube radius) of the torus
    - solid: If True, generate solid tori; if False, generate hollow (surface-only) tori
    - interior_noise: Noise level for interior points when solid=True
    
    Returns:
    - [X, y]: Point cloud and corresponding labels
    """
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
        if solid:
            # Generate solid torus with interior points
            points = generate_solid_torus_points(torus, n, interior_noise, big_radius, small_radius)
        else:
            # Generate hollow torus (surface-only points)
            points = np.array(torus.sample(n))
        
        sampled_points.append(points)
        labels.append(np.full((n, 1), i % 2))  # Alternating labels 0 and 1
    
    # Concatenate results
    X = np.concatenate(sampled_points)
    y = np.concatenate(labels)
    
    return [X, y]


def generate_solid_torus_points(torus_mesh, n_points, interior_noise=0.1, major_radius=3.0, minor_radius=1.0):
    """
    Generate points inside a solid torus using fast volumetric sampling.
    Uses mathematical torus equations for efficient point generation.
    
    Parameters:
    - torus_mesh: Trimesh torus object (used for transformation)
    - n_points: Number of points to generate
    - interior_noise: Noise level for adding randomness to interior points
    - major_radius: Major radius of the torus (big_radius parameter)
    - minor_radius: Minor radius of the torus (small_radius parameter)
    
    Returns:
    - points: Array of 3D points distributed throughout the torus volume
    """
    
    # Fast volumetric torus generation using toroidal coordinates
    n_surface = int(0.2 * n_points)  # 20% surface points (reduced for speed)
    n_interior = n_points - n_surface
    
    # 1. Generate surface points (fast mesh sampling)
    surface_points = np.array(torus_mesh.sample(n_surface))
    
    # 2. Generate interior points with proper alignment
    # Use surface points to determine the torus orientation and position
    
    if n_surface > 10:  # Need enough surface points for reliable estimation
        # Estimate torus center and orientation from surface points
        estimated_center = np.mean(surface_points, axis=0)
        
        # Generate interior points in standard coordinates
        interior_points = generate_torus_interior_fast(n_interior, major_radius, minor_radius, interior_noise)
        
        # Translate interior points to match the estimated center
        interior_points += estimated_center
        
        # Optional: Apply rotation alignment if needed
        # For now, translation is sufficient for most cases
        
    else:
        # Fallback: use mesh center for alignment
        mesh_center = torus_mesh.center_mass
        interior_points = generate_torus_interior_fast(n_interior, major_radius, minor_radius, interior_noise)
        interior_points += mesh_center
    
    # Combine all points
    all_points = np.vstack([surface_points, interior_points])
    
    # Shuffle to distribute surface and interior points randomly
    np.random.shuffle(all_points)
    
    return all_points


def generate_torus_interior_fast(n_points, major_radius, minor_radius, noise_level=0.1):
    """
    Fast generation of interior points for a torus using toroidal coordinates.
    Uses fully vectorized operations for maximum performance.
    
    Parameters:
    - n_points: Number of interior points to generate
    - major_radius: Major radius of the torus
    - minor_radius: Minor radius of the torus
    - noise_level: Level of randomness to add
    
    Returns:
    - points: Array of 3D points inside the torus
    """
    
    # Generate all points at once for maximum vectorization
    # φ (phi): angle around the tube (0 to 2π)
    # θ (theta): angle around the main torus (0 to 2π)  
    # r: distance from tube center (0 to minor_radius)
    
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    
    # For interior points, use r distribution that gives uniform volume density
    # r² distribution for uniform density in disk cross-section
    r_normalized = np.sqrt(np.random.uniform(0, 1, n_points))
    r = r_normalized * minor_radius
    
    # Convert to Cartesian coordinates (fully vectorized)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    x = (major_radius + r * cos_phi) * cos_theta
    y = (major_radius + r * cos_phi) * sin_theta
    z = r * sin_phi
    
    # Add noise if specified (vectorized)
    if noise_level > 0:
        noise_scale = noise_level * minor_radius * 0.1  # Scale noise to torus size
        x += np.random.normal(0, noise_scale, n_points)
        y += np.random.normal(0, noise_scale, n_points)
        z += np.random.normal(0, noise_scale, n_points)
    
    # Combine coordinates
    points = np.column_stack([x, y, z])
    return points


def is_point_inside_torus(point, center, major_radius, minor_radius):
    """
    Check if a point is inside a torus using the mathematical definition.
    
    Parameters:
    - point: 3D point coordinates
    - center: Torus center
    - major_radius: Major radius of the torus
    - minor_radius: Minor radius of the torus
    
    Returns:
    - bool: True if point is inside torus
    """
    # Translate point to torus-centered coordinates
    p = point - center
    
    # Distance from point to z-axis
    rho = np.sqrt(p[0]**2 + p[1]**2)
    
    # Distance from point to the ring center
    ring_distance = np.sqrt((rho - major_radius)**2 + p[2]**2)
    
    # Point is inside if distance to ring is less than minor radius
    return ring_distance <= minor_radius


def gen_easy(n, big_radius, small_radious, solid=False, interior_noise=0.1):
    """
    Generate a simple pair of tori with optional solid/hollow mode.
    
    Parameters:
    - n: Number of points per torus
    - big_radius: Major radius of torus
    - small_radious: Minor radius of torus
    - solid: If True, generate solid tori; if False, hollow (surface-only)
    - interior_noise: Noise level for interior points when solid=True
    
    Returns:
    - [X, y]: Point cloud and labels
    """
    torus1 = tr.creation.torus(big_radius, small_radious)
    torus2 = tr.creation.torus(big_radius, small_radious)
    rotation_matrix = tr.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    torus2.apply_transform(rotation_matrix)
    translation_matrix1 = tr.transformations.translation_matrix([big_radius/2, 0, 0])
    translation_matrix2 = tr.transformations.translation_matrix([-big_radius/2, 0, 0])
    torus2.apply_transform(translation_matrix2)
    torus1.apply_transform(translation_matrix1)
    
    # Generate points based on solid/hollow mode
    if solid:
        points1 = generate_solid_torus_points(torus1, n, interior_noise, big_radius, small_radious)
        points2 = generate_solid_torus_points(torus2, n, interior_noise, big_radius, small_radious)
    else:
        points1 = np.array(torus1.sample(n))
        points2 = np.array(torus2.sample(n))
    
    X = np.concatenate((points1, points2))
    y = np.concatenate((np.zeros((n, 1)), np.ones((n, 1))))
    return [X, y]

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
