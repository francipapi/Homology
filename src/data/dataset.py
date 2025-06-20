import trimesh as tr 
import numpy as np
import torch
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

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


def generate_solid_torus_points(
        torus_mesh: tr.Trimesh,
        n_points: int,
        interior_noise: float = 0.1,
        major_radius: float = 3.0,
        minor_radius: float = 1.0,
):
    """
    Volumetrically sample `n_points` inside a **solid** torus.

    • Works for any translation / rotation you applied to the mesh  
    • No expensive `mesh.contains()` calls – 100 % acceptance, O(n) time  
    • Still guarantees every point is inside the torus (even after noise)

    Parameters
    ----------
    torus_mesh : trimesh.Trimesh
        The torus mesh **after** all transforms.
    n_points : int
        Number of interior samples.
    interior_noise : float, optional
        σ of zero-mean Gaussian noise as a fraction of `minor_radius`.
    major_radius, minor_radius : float, optional
        The torus R and r used when you built the mesh; needed only once
        to generate the canonical samples.

    Returns
    -------
    (n_points, 3) ndarray[float]
        Uniform volumetric samples in world coordinates.
    """

    # ---------- 1. Fast local-to-world frame extraction --------------------
    # Use the oriented bounding box (OBB) as a cheap way to recover the
    # torus centre and orientation 
    obb              = torus_mesh.bounding_box_oriented
    T_world_from_obb = obb.primitive.transform          # 4×4 – R|t
    centre           = T_world_from_obb[:3, 3]
    R_world_from_obb = T_world_from_obb[:3, :3]

    # Smallest OBB extent ⇒ tube axis direction (Z in the canonical frame)
    axis_small  = int(np.argmin(obb.primitive.extents))
    # Re-order the rotation columns so that canonical ẑ → tube-axis
    perm        = [0, 1, 2]
    perm.remove(axis_small)
    perm.append(axis_small)
    R_world_from_local = R_world_from_obb[:, perm]      # 3×3

    # ---------- 2. Canonical, **rejection-free** torus volume sampling ----
    #   θ  ∈ [0, 2π) - around the major circle
    #   φ  ∈ [0, 2π) - within the tube cross-section
    #   ρ² ∈ [0, r²] – radius in the tube disc (√U trick for uniform area)
    theta = np.random.rand(n_points) * 2.0 * np.pi
    phi   = np.random.rand(n_points) * 2.0 * np.pi
    rho   = np.sqrt(np.random.rand(n_points)) * minor_radius

    cosθ, sinθ = np.cos(theta), np.sin(theta)
    cosφ, sinφ = np.cos(phi),   np.sin(phi)

    x_local = (major_radius + rho * cosφ) * cosθ
    y_local = (major_radius + rho * cosφ) * sinθ
    z_local = rho * sinφ
    local_pts = np.stack((x_local, y_local, z_local), axis=1)   # (n,3)

    # ---------- 3. Transform to the mesh’s world space --------------------
    world_pts = local_pts @ R_world_from_local.T + centre

    # ---------- 4. Optional interior jitter (remains inside) --------------
    if interior_noise > 0.0:
        sigma   = interior_noise * minor_radius * 0.05
        noise   = np.random.normal(0.0, sigma, size=world_pts.shape)
        cand    = world_pts + noise

        # Keep only noisy candidates that are still inside analytically
        axis_vec   = R_world_from_local[:, 2]                 # tube axis
        pc         = cand - centre
        z_coord    = pc @ axis_vec
        radial_vec = pc - np.outer(z_coord, axis_vec)
        rho_val    = np.linalg.norm(radial_vec, axis=1)
        inside_ok  = (rho_val - major_radius) ** 2 + z_coord ** 2 <= minor_radius ** 2

        world_pts[inside_ok] = cand[inside_ok]

    return world_pts




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
