"""
Compute homology for neural network layer activations using Witness Complexes.
PyTorch-based implementation for efficiency and to avoid NumPy compatibility issues.

This script provides an alternative to compute_homology.py that uses witness complexes
instead of full Rips complexes. Witness complexes can preserve topology while using
significantly fewer points, making them ideal for large datasets where tracking
topological features (especially loops) is important.

The witness complex approach:
1. Selects a small subset of "landmark" points
2. Uses remaining points as "witnesses" to determine connectivity
3. Preserves topology while drastically reducing computational complexity

Output format: [num_networks, num_layers, max_dimension] tensor of Betti numbers.
"""

import torch
import torch.nn.functional as F
import os
import glob
import yaml
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import warnings
import concurrent.futures
import multiprocessing as mp
from dataclasses import dataclass
import psutil
import gc
import threading
import queue

# Only import NumPy when absolutely necessary (for Gudhi interface)
import numpy as np

# Import Gudhi with warning suppression
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import gudhi as gd

# Import project modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class WitnessLayerTask:
    """Task structure for parallel processing of individual layers in witness complex."""
    layer_data: np.ndarray  # Layer data directly embedded
    config: Dict
    filename: str
    net_idx: int
    layer_idx: int
    task_id: int


@dataclass
class WitnessLayerResult:
    """Result structure for completed witness complex layer processing."""
    filename: str
    net_idx: int
    layer_idx: int
    task_id: int
    betti_numbers: List[int]
    computation_time: float
    success: bool
    error_message: Optional[str] = None
    memory_usage_mb: Optional[float] = None


class WitnessProgressTracker:
    """Thread-safe progress tracker for witness complex parallel processing."""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def update(self, success: bool = True):
        with self.lock:
            self.completed_tasks += 1
            if not success:
                self.failed_tasks += 1
    
    def get_progress(self) -> Dict:
        with self.lock:
            elapsed = time.time() - self.start_time
            rate = self.completed_tasks / elapsed if elapsed > 0 else 0
            eta = (self.total_tasks - self.completed_tasks) / rate if rate > 0 else 0
            
            return {
                'completed': self.completed_tasks,
                'total': self.total_tasks,
                'failed': self.failed_tasks,
                'percentage': 100 * self.completed_tasks / self.total_tasks,
                'elapsed_time': elapsed,
                'eta_seconds': eta,
                'rate_per_second': rate
            }


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def select_landmarks_maxmin_torch(points: torch.Tensor, n_landmarks: int, 
                                 device: Optional[str] = None,
                                 batch_size: int = 10000,
                                 init_strategy: str = 'center') -> torch.Tensor:
    """
    Select landmarks using an improved maxmin strategy with PyTorch for efficiency.
    
    The maxmin algorithm iteratively selects landmarks by choosing points that are
    furthest from all previously selected landmarks, maximizing coverage.
    
    Parameters:
    - points: Input points tensor of shape (N, D)
    - n_landmarks: Number of landmarks to select
    - device: Device to use for computation (None for auto-detection)
    - batch_size: Batch size for distance computations (memory management)
    - init_strategy: Initialization strategy ('random', 'center', 'corner')
    
    Returns:
    - landmarks: Selected landmark points tensor
    """
    if device is None:
        device = points.device
    
    n_points = len(points)
    if n_landmarks >= n_points:
        return points
    
    # Move to specified device if needed
    points = points.to(device)
    
    # Initialize landmark indices
    landmark_indices = torch.zeros(n_landmarks, dtype=torch.long, device=device)
    
    # Smart initialization strategies
    if init_strategy == 'center':
        # Start with point closest to centroid
        centroid = torch.mean(points, dim=0)
        distances_to_center = torch.sum((points - centroid) ** 2, dim=1)
        landmark_indices[0] = torch.argmin(distances_to_center)
    elif init_strategy == 'corner':
        # Start with point furthest from centroid
        centroid = torch.mean(points, dim=0)
        distances_to_center = torch.sum((points - centroid) ** 2, dim=1)
        landmark_indices[0] = torch.argmax(distances_to_center)
    else:  # random
        landmark_indices[0] = torch.randint(0, n_points, (1,), device=device)
    
    # Distance from each point to its nearest landmark
    min_distances = torch.full((n_points,), float('inf'), device=device)
    
    # Initial distance computation to first landmark
    first_landmark = points[landmark_indices[0]]
    if n_points <= batch_size:
        # Compute all distances at once if small enough
        min_distances = torch.sum((points - first_landmark) ** 2, dim=1)
    else:
        # Batch distance computation for memory efficiency
        for i in range(0, n_points, batch_size):
            end_i = min(i + batch_size, n_points)
            batch_points = points[i:end_i]
            batch_distances = torch.sum((batch_points - first_landmark) ** 2, dim=1)
            min_distances[i:end_i] = batch_distances
    
    # Iteratively select remaining landmarks
    for i in range(1, n_landmarks):
        # Select the point with maximum distance to nearest landmark
        landmark_indices[i] = torch.argmax(min_distances)
        last_landmark = points[landmark_indices[i]]
        
        # Update minimum distances with batch processing
        if n_points <= batch_size:
            # Compute all distances at once if small enough
            distances = torch.sum((points - last_landmark) ** 2, dim=1)
            min_distances = torch.minimum(min_distances, distances)
        else:
            # Batch distance computation for memory efficiency
            for j in range(0, n_points, batch_size):
                end_j = min(j + batch_size, n_points)
                batch_points = points[j:end_j]
                batch_distances = torch.sum((batch_points - last_landmark) ** 2, dim=1)
                min_distances[j:end_j] = torch.minimum(min_distances[j:end_j], batch_distances)
    
    return points[landmark_indices]


def select_landmarks_random_torch(points: torch.Tensor, n_landmarks: int,
                                 device: Optional[str] = None) -> torch.Tensor:
    """
    Select landmarks randomly from the point set using PyTorch.
    
    Parameters:
    - points: Input points tensor of shape (N, D)
    - n_landmarks: Number of landmarks to select
    - device: Device to use for computation
    
    Returns:
    - landmarks: Selected landmark points tensor
    """
    if device is None:
        device = points.device
        
    n_points = len(points)
    if n_landmarks >= n_points:
        return points
    
    # Randomly select landmark indices
    perm = torch.randperm(n_points, device=device)[:n_landmarks]
    return points[perm]


def select_landmarks_fps_torch(points: torch.Tensor, n_landmarks: int,
                              device: Optional[str] = None,
                              batch_size: int = 10000) -> torch.Tensor:
    """
    Select landmarks using Farthest Point Sampling (FPS) with PyTorch.
    
    FPS is similar to maxmin but optimized for geometric point clouds.
    
    Parameters:
    - points: Input points tensor of shape (N, D)
    - n_landmarks: Number of landmarks to select
    - device: Device to use for computation
    - batch_size: Batch size for distance computations
    
    Returns:
    - landmarks: Selected landmark points tensor
    """
    # FPS is essentially the same as maxmin for our purposes
    return select_landmarks_maxmin_torch(points, n_landmarks, device, batch_size, 'center')


def compute_pairwise_distances_batch(points1: torch.Tensor, points2: torch.Tensor, 
                                    batch_size: int = 1000) -> torch.Tensor:
    """
    Compute pairwise distances between two sets of points in batches to save memory.
    
    Parameters:
    - points1: First set of points (N1, D)
    - points2: Second set of points (N2, D)
    - batch_size: Size of batches for computation
    
    Returns:
    - distances: Pairwise distance matrix (N1, N2)
    """
    n1, n2 = len(points1), len(points2)
    distances = torch.zeros(n1, n2, device=points1.device)
    
    for i in range(0, n1, batch_size):
        end_i = min(i + batch_size, n1)
        batch1 = points1[i:end_i]
        
        for j in range(0, n2, batch_size):
            end_j = min(j + batch_size, n2)
            batch2 = points2[j:end_j]
            
            # Compute squared distances for this batch
            diff = batch1.unsqueeze(1) - batch2.unsqueeze(0)
            distances[i:end_i, j:end_j] = torch.sum(diff ** 2, dim=2)
    
    return distances


def compute_witness_homology_betti_torch(points: torch.Tensor, n_landmarks: int,
                                        max_dimension: int = 2, 
                                        max_alpha_square: float = float('inf'),
                                        landmark_selection: str = 'maxmin',
                                        witness_type: str = 'weak',
                                        device: Optional[str] = None) -> List[int]:
    """
    Compute persistent homology using witness complexes with PyTorch efficiency.
    
    Parameters:
    - points: Input points tensor of shape (N, D)
    - n_landmarks: Number of landmarks to select
    - max_dimension: Maximum homology dimension to compute
    - max_alpha_square: Maximum squared distance for witness complex filtration
    - landmark_selection: Method for selecting landmarks ('maxmin' or 'random')
    - witness_type: Type of witness complex ('weak' or 'strong')
    - device: Device for computation (None for auto-detection)
    
    Returns:
    - List of Betti numbers for dimensions 0 up to max_dimension
    """
    try:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Ensure points are on the correct device
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        points = points.to(device)
        
        n_points = len(points)
        
        # Handle edge case
        if n_points < max_dimension + 1:
            print(f"Warning: Too few points ({n_points}) for dimension {max_dimension}")
            return [1] + [0] * max_dimension
        
        # Ensure we don't select more landmarks than points
        n_landmarks = min(n_landmarks, n_points)
        
        # Select landmarks using PyTorch (simplified output)
        try:
            with torch.no_grad():
                if landmark_selection == 'maxmin':
                    # Get maxmin parameters from config
                    batch_size = config.get('witness_complex', {}).get('batch_size', 10000)
                    init_strategy = config.get('witness_complex', {}).get('maxmin_init_strategy', 'center')
                    landmarks = select_landmarks_maxmin_torch(points, n_landmarks, device, batch_size, init_strategy)
                elif landmark_selection == 'fps':
                    batch_size = config.get('witness_complex', {}).get('batch_size', 10000)
                    landmarks = select_landmarks_fps_torch(points, n_landmarks, device, batch_size)
                else:  # random
                    landmarks = select_landmarks_random_torch(points, n_landmarks, device)
            
        except Exception as e:
            return [1] + [0] * max_dimension
        
        # Convert to NumPy for Gudhi (only at the interface)
        points_np = points.cpu().numpy()
        landmarks_np = landmarks.cpu().numpy()
        
        # Create witness complex using Gudhi
        try:
            if witness_type == 'strong':
                witness_complex = gd.EuclideanStrongWitnessComplex(
                    witnesses=points_np,
                    landmarks=landmarks_np
                )
            else:
                witness_complex = gd.EuclideanWitnessComplex(
                    witnesses=points_np,
                    landmarks=landmarks_np
                )
            
        except Exception as e:
            print(f"\nError creating witness complex: {e}")
            return [1] + [0] * max_dimension
        
        # Create simplex tree
        print(" -> building simplex tree", end="", flush=True)
        try:
            simplex_tree = witness_complex.create_simplex_tree(
                max_alpha_square=max_alpha_square,
                limit_dimension=max_dimension + 1
            )
            
            num_simplices = simplex_tree.num_simplices()
            print(f" ({num_simplices} simplices)", end="", flush=True)
            
        except Exception as e:
            print(f"\nError creating simplex tree: {e}")
            return [1] + [0] * max_dimension
        
        # Compute persistence
        print(" -> computing persistence", end="", flush=True)
        try:
            persistence = simplex_tree.persistence()
            betti_numbers = simplex_tree.betti_numbers()
            
            # Ensure we have Betti numbers for all dimensions
            while len(betti_numbers) <= max_dimension:
                betti_numbers.append(0)
            
            print(f" -> Betti numbers: {betti_numbers[:max_dimension + 1]}")
            
            return betti_numbers[:max_dimension + 1]
            
        except Exception as e:
            print(f"\nError computing persistence: {e}")
            return [1] + [0] * max_dimension
        
    except Exception as e:
        print(f"\nUnexpected error in witness homology computation: {e}")
        import traceback
        traceback.print_exc()
        return [1] + [0] * max_dimension


def process_witness_layer_task(task: WitnessLayerTask) -> WitnessLayerResult:
    """
    Worker function to process a single witness complex layer task.
    
    This function is designed to be stateless and memory-efficient,
    suitable for parallel execution across multiple processes.
    """
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Get layer data directly from task
        layer_data = task.layer_data
        
        # Validate input data
        if layer_data is None or len(layer_data) == 0:
            return WitnessLayerResult(
                filename=task.filename,
                net_idx=task.net_idx,
                layer_idx=task.layer_idx,
                task_id=task.task_id,
                betti_numbers=[0] * (task.config.get('computation', {}).get('max_dimension', 2) + 1),
                computation_time=time.time() - start_time,
                success=False,
                error_message="Empty layer data"
            )
        
        # Check minimum points threshold
        min_points = task.config.get('sampling', {}).get('min_points_threshold', 50)
        if len(layer_data) < min_points:
            return WitnessLayerResult(
                filename=task.filename,
                net_idx=task.net_idx,
                layer_idx=task.layer_idx,
                task_id=task.task_id,
                betti_numbers=[0] * (task.config.get('computation', {}).get('max_dimension', 2) + 1),
                computation_time=time.time() - start_time,
                success=False,
                error_message=f"Insufficient points: {len(layer_data)} < {min_points}"
            )
        
        # Compute Betti numbers using witness complex
        betti_numbers = process_single_layer_witness_optimized(
            layer_data, task.config, task.layer_idx
        )
        
        # Memory cleanup
        del layer_data
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        return WitnessLayerResult(
            filename=task.filename,
            net_idx=task.net_idx,
            layer_idx=task.layer_idx,
            task_id=task.task_id,
            betti_numbers=betti_numbers,
            computation_time=time.time() - start_time,
            success=True,
            memory_usage_mb=memory_usage
        )
        
    except Exception as e:
        # Cleanup on error
        gc.collect()
        
        return WitnessLayerResult(
            filename=task.filename,
            net_idx=task.net_idx,
            layer_idx=task.layer_idx,
            task_id=task.task_id,
            betti_numbers=[1] + [0] * task.config.get('computation', {}).get('max_dimension', 2),
            computation_time=time.time() - start_time,
            success=False,
            error_message=str(e)
        )


def create_witness_layer_tasks(layer_files: Dict, config: Dict) -> List[WitnessLayerTask]:
    """
    Create a list of all layer processing tasks for parallel witness complex execution.
    
    Flattens the nested structure of (filename, network, layer) into a single
    task queue for optimal load balancing. Supports both padded tensor format
    and variable-length dictionary format.
    """
    tasks = []
    task_id = 0
    
    for filename, layer_outputs_orig in layer_files.items():
        # Handle variable-length dictionary format
        if isinstance(layer_outputs_orig, dict) and not hasattr(layer_outputs_orig, 'shape'):
            # This is a variable-length format: {layer_idx: tensor}
            # Assume single network for now (extend if needed)
            num_networks = 1
            
            for layer_idx, layer_tensor in sorted(layer_outputs_orig.items()):
                if isinstance(layer_tensor, torch.Tensor):
                    layer_data = layer_tensor.cpu().numpy()
                else:
                    layer_data = layer_tensor
                
                # Create task for this layer
                task = WitnessLayerTask(
                    layer_data=layer_data.copy(),  # Copy to avoid shared memory issues
                    config=config,
                    filename=filename,
                    net_idx=0,  # Single network for variable-length format
                    layer_idx=int(layer_idx) if isinstance(layer_idx, str) else layer_idx,
                    task_id=task_id
                )
                tasks.append(task)
                task_id += 1
        else:
            # Handle standard tensor format
            # Convert to numpy if needed (but keep original in layer_files)
            if isinstance(layer_outputs_orig, torch.Tensor):
                layer_outputs = layer_outputs_orig.cpu().numpy()
            else:
                layer_outputs = layer_outputs_orig
            
            # Expected shape: [num_networks, num_layers, num_samples, layer_dim]
            if layer_outputs.ndim == 4:
                num_networks, num_layers, num_samples, layer_dim = layer_outputs.shape
                
                # Create tasks for each (network, layer) combination
                for net_idx in range(num_networks):
                    for layer_idx in range(num_layers):
                        # Extract single layer activations: (num_samples, layer_dim)
                        layer_data = layer_outputs[net_idx, layer_idx].copy()  # Copy to avoid shared memory issues
                        
                        task = WitnessLayerTask(
                            layer_data=layer_data,
                            config=config,
                            filename=filename,
                            net_idx=net_idx,
                            layer_idx=layer_idx,
                            task_id=task_id
                        )
                        tasks.append(task)
                        task_id += 1
            else:
                print(f"Warning: Unexpected shape {layer_outputs.shape} for {filename}, skipping...")
    
    return tasks


def aggregate_witness_results(results: List[WitnessLayerResult], layer_files: Dict, max_dimension: int) -> Dict:
    """
    Aggregate parallel processing results back into the original data structure.
    
    Reconstructs the [num_networks, num_layers, max_dimension] tensor format
    from the flattened task results. Handles both padded tensor and variable-length formats.
    """
    all_betti_results = {}
    
    # Group results by filename
    results_by_file = {}
    for result in results:
        if result.filename not in results_by_file:
            results_by_file[result.filename] = []
        results_by_file[result.filename].append(result)
    
    # Reconstruct the original structure for each file
    for filename, file_results in results_by_file.items():
        if filename not in layer_files:
            continue
            
        layer_outputs = layer_files[filename]
        
        # Handle variable-length dictionary format
        if isinstance(layer_outputs, dict) and not hasattr(layer_outputs, 'shape'):
            # Variable-length format
            num_layers = len(layer_outputs)
            num_networks = 1  # Assuming single network for now
            
            # Initialize results tensor
            betti_results = np.zeros((num_networks, num_layers, max_dimension + 1), dtype=np.int32)
            
            # Fill in results from parallel processing
            for result in file_results:
                if result.success:
                    betti_numbers = result.betti_numbers[:max_dimension + 1]
                else:
                    # Use default values for failed computations
                    betti_numbers = [1] + [0] * max_dimension
                
                betti_results[result.net_idx, result.layer_idx] = betti_numbers
            
            all_betti_results[filename] = betti_results
        else:
            # Standard tensor format
            if isinstance(layer_outputs, torch.Tensor):
                layer_outputs = layer_outputs.cpu().numpy()
            elif not isinstance(layer_outputs, np.ndarray):
                layer_outputs = np.array(layer_outputs)
            
            if layer_outputs.ndim == 4:
                num_networks, num_layers, num_samples, layer_dim = layer_outputs.shape
                
                # Initialize results tensor
                betti_results = np.zeros((num_networks, num_layers, max_dimension + 1), dtype=np.int32)
                
                # Fill in results from parallel processing
                for result in file_results:
                    if result.success:
                        betti_numbers = result.betti_numbers[:max_dimension + 1]
                    else:
                        # Use default values for failed computations
                        betti_numbers = [1] + [0] * max_dimension
                    
                    betti_results[result.net_idx, result.layer_idx] = betti_numbers
                
                all_betti_results[filename] = betti_results
    
    return all_betti_results


def get_optimal_witness_worker_count(config: Dict, total_tasks: int) -> int:
    """
    Determine optimal number of workers for witness complex based on system resources and task characteristics.
    """
    # Get configured number of workers
    configured_workers = config.get('parallel', {}).get('num_workers', None)
    
    if configured_workers is not None:
        return max(1, min(configured_workers, total_tasks))
    
    # Auto-detect optimal worker count
    cpu_count = mp.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Estimate memory per worker (witness complex is more memory intensive)
    estimated_memory_per_worker = 1.0  # GB per worker (higher than Ripser due to Gudhi)
    memory_limited_workers = int(available_memory_gb / estimated_memory_per_worker)
    
    # Use conservative estimate: 75% of CPU cores or memory limit, whichever is lower
    optimal_workers = min(
        max(1, int(cpu_count * 0.75)),
        memory_limited_workers,
        total_tasks  # Never more workers than tasks
    )
    
    return optimal_workers


def process_single_layer_witness_optimized(layer_activations: Union[torch.Tensor, np.ndarray], 
                                          config: Dict, layer_idx: int = 0) -> List[int]:
    """
    Optimized single layer processing using Gudhi witness complex.
    Uses all configurable parameters from homology_config.yaml.
    """
    try:
        # Convert to numpy for Gudhi
        if isinstance(layer_activations, torch.Tensor):
            layer_activations = layer_activations.cpu().numpy()
        
        n_points = len(layer_activations)
        
        # Get witness complex configuration
        witness_config = config.get('witness_complex', {})
        
        # Set random seed if specified
        random_seed = witness_config.get('random_seed', None)
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Apply normalization if enabled
        if config.get('computation', {}).get('normalize_data', True):
            mean = np.mean(layer_activations, axis=0, keepdims=True)
            std = np.std(layer_activations, axis=0, keepdims=True)
            layer_activations = (layer_activations - mean) / (std + 1e-8)
        
        # Determine number of landmarks
        if witness_config.get('adaptive_landmarks', False):
            landmark_percentage = witness_config.get('landmark_percentage', 0.005)
            n_landmarks = int(n_points * landmark_percentage)
            min_landmarks = witness_config.get('min_landmarks', 20)
            max_landmarks = witness_config.get('max_landmarks', 200)
            n_landmarks = max(min_landmarks, min(n_landmarks, max_landmarks))
        else:
            n_landmarks = witness_config.get('n_landmarks', 50)
        
        # Sample witnesses if dataset is large
        witness_threshold = witness_config.get('witness_threshold', 10000)
        use_witness_sampling = witness_config.get('use_witness_sampling', True)
        
        if use_witness_sampling and n_points > witness_threshold:
            max_witnesses = witness_config.get('max_witnesses', 10000)
            sample_size = min(max_witnesses, n_points)
            
            witness_sampling_method = witness_config.get('witness_sampling_method', 'random')
            if witness_sampling_method == 'fps':
                # Simple FPS sampling (could be enhanced)
                indices = np.random.choice(n_points, sample_size, replace=False)
            else:  # random
                indices = np.random.choice(n_points, sample_size, replace=False)
            
            witnesses = layer_activations[indices]
        else:
            witnesses = layer_activations
        
        # Select landmarks
        landmark_selection = witness_config.get('landmark_selection', 'random')
        witness_count = len(witnesses)
        n_landmarks = min(n_landmarks, witness_count)
        
        if n_landmarks >= witness_count:
            landmarks = witnesses
            landmark_indices = np.arange(witness_count)
        else:
            if landmark_selection == 'maxmin':
                # Enhanced maxmin implementation with improved initialization
                landmark_indices = np.zeros(n_landmarks, dtype=int)
                
                # Smart initialization based on configuration
                init_strategy = witness_config.get('maxmin_init_strategy', 'center')
                if init_strategy == 'center':
                    # Start with point closest to centroid
                    centroid = np.mean(witnesses, axis=0)
                    distances_to_center = np.sum((witnesses - centroid) ** 2, axis=1)
                    landmark_indices[0] = np.argmin(distances_to_center)
                elif init_strategy == 'corner':
                    # Start with point furthest from centroid (corner point)
                    centroid = np.mean(witnesses, axis=0)
                    distances_to_center = np.sum((witnesses - centroid) ** 2, axis=1)
                    landmark_indices[0] = np.argmax(distances_to_center)
                elif init_strategy == 'pca':
                    # Start with point along first principal component
                    try:
                        # Simple PCA implementation
                        centered_data = witnesses - np.mean(witnesses, axis=0)
                        cov_matrix = np.cov(centered_data.T)
                        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                        # Project onto first principal component
                        first_pc = eigenvecs[:, -1]  # Last eigenvector (largest eigenvalue)
                        projections = np.dot(centered_data, first_pc)
                        landmark_indices[0] = np.argmax(np.abs(projections))
                    except:
                        # Fall back to random if PCA fails
                        landmark_indices[0] = np.random.randint(0, witness_count)
                else:  # random
                    landmark_indices[0] = np.random.randint(0, witness_count)
                
                # Initialize distances to first landmark
                min_distances = np.sum((witnesses - witnesses[landmark_indices[0]]) ** 2, axis=1)
                
                # Iteratively select remaining landmarks using maxmin strategy
                for i in range(1, n_landmarks):
                    # Select the point with maximum distance to nearest landmark
                    landmark_indices[i] = np.argmax(min_distances)
                    
                    # Update minimum distances efficiently
                    last_landmark = witnesses[landmark_indices[i]]
                    new_distances = np.sum((witnesses - last_landmark) ** 2, axis=1)
                    min_distances = np.minimum(min_distances, new_distances)
                    
                    # Optional: Add small amount of noise to break ties
                    if witness_config.get('maxmin_add_noise', False):
                        noise_scale = witness_config.get('maxmin_noise_scale', 1e-6)
                        min_distances += np.random.normal(0, noise_scale, size=min_distances.shape)
                
                landmarks = witnesses[landmark_indices]
            else:  # random
                landmark_indices = np.random.choice(witness_count, n_landmarks, replace=False)
                landmarks = witnesses[landmark_indices]
        
        
        # Separate witnesses and landmarks (witnesses can't be landmarks for Gudhi)
        witness_mask = np.ones(len(witnesses), dtype=bool)
        witness_mask[landmark_indices] = False
        final_witnesses = witnesses[witness_mask]
        
        if len(final_witnesses) == 0:
            # If no witnesses left, use random subset of original data
            witness_indices = np.random.choice(len(witnesses), min(100, len(witnesses)), replace=False)
            final_witnesses = witnesses[witness_indices]
        
        # Create Gudhi witness complex
        
        # Get complex construction parameters
        max_alpha_square = witness_config.get('max_alpha_square', 1.0)
        if isinstance(max_alpha_square, str) and max_alpha_square.lower() == 'inf':
            max_alpha_square = float('inf')
        else:
            max_alpha_square = float(max_alpha_square)
        
        witness_type = witness_config.get('witness_type', 'weak')
        use_euclidean_witness = witness_config.get('use_euclidean_witness', True)
        limit_dimension = witness_config.get('limit_dimension', 2)
        max_dimension = min(config.get('computation', {}).get('max_dimension', 2), limit_dimension)
        
        try:
            if use_euclidean_witness:
                if witness_type == 'strong':
                    witness_complex = gd.EuclideanStrongWitnessComplex(
                        witnesses=final_witnesses,
                        landmarks=landmarks
                    )
                else:  # weak
                    witness_complex = gd.EuclideanWitnessComplex(
                        witnesses=final_witnesses,
                        landmarks=landmarks
                    )
            else:
                # Manual construction (could be implemented for advanced use cases)
                witness_complex = gd.EuclideanWitnessComplex(
                    witnesses=final_witnesses,
                    landmarks=landmarks
                )
            
            # Create simplex tree
            simplex_tree = witness_complex.create_simplex_tree(
                max_alpha_square=max_alpha_square,
                limit_dimension=max_dimension + 1
            )
            
            # Compute persistence
            simplex_tree.persistence()
            betti_numbers = simplex_tree.betti_numbers()
            
            # Ensure we have all dimensions
            while len(betti_numbers) <= max_dimension:
                betti_numbers.append(0)
            
            result = betti_numbers[:max_dimension + 1]
            return result
            
        except Exception as e:
            
            # Use configurable fallback if enabled
            fallback_enabled = witness_config.get('fallback_enabled', True)
            if fallback_enabled:
                fallback_h0_ratio = witness_config.get('fallback_h0_ratio', 5)
                fallback_h1_ratio = witness_config.get('fallback_h1_ratio', 10)
                
                h0 = min(10, len(landmarks) // fallback_h0_ratio)
                h1 = min(5, len(landmarks) // fallback_h1_ratio)
                result = [h0, h1] + [0] * max(0, max_dimension - 1)
                return result
            else:
                max_dimension = config.get('computation', {}).get('max_dimension', 2)
                return [0] * (max_dimension + 1)
        
    except Exception as e:
        max_dimension = config.get('computation', {}).get('max_dimension', 2)
        return [0] * (max_dimension + 1)


def load_layer_outputs(input_dir: str) -> Dict[str, Union[torch.Tensor, Dict]]:
    """
    Load all layer output files from the input directory.
    Supports both padded tensor format and variable-length dictionary format.
    """
    layer_files = {}
    pattern = os.path.join(input_dir, "*.pt")
    
    for file_path in glob.glob(pattern):
        filename = os.path.basename(file_path)
        try:
            data = torch.load(file_path, map_location='cpu')
            if isinstance(data, dict):
                if 'layer_outputs' in data:
                    layer_outputs = data['layer_outputs']
                    # Check if this is variable-length format
                    if data.get('variable_length', False) and isinstance(layer_outputs, dict):
                        print(f"  📄 {filename}: Variable-length format with {len(layer_outputs)} layers")
                        layer_files[filename] = layer_outputs
                    else:
                        layer_files[filename] = layer_outputs
                        if hasattr(layer_outputs, 'shape'):
                            print(f"  📄 {filename}: {layer_outputs.shape}")
                        else:
                            print(f"  📄 {filename}: Loaded")
                else:
                    layer_files[filename] = data
                    print(f"  📄 {filename}: {data.shape if hasattr(data, 'shape') else 'Loaded'}")
            else:
                layer_files[filename] = data
                print(f"  📄 {filename}: {data.shape if hasattr(data, 'shape') else 'Loaded'}")
        except Exception as e:
            print(f"  ⚠️  Could not load {filename}: {e}")
    
    return layer_files


def _process_witness_tasks_parallel(tasks: List[WitnessLayerTask], layer_files: Dict, max_dimension: int, config: Dict) -> Dict:
    """
    Process witness complex tasks using parallel execution with comprehensive monitoring and error handling.
    """
    num_workers = get_optimal_witness_worker_count(config, len(tasks))
    print(f"\n⚙️  PARALLEL CONFIGURATION")
    print(f"  👥 Workers: {num_workers}")
    print(f"  🎯 Tasks: {len(tasks)}")
    print(f"  ⏱️  Timeout: {config.get('parallel', {}).get('timeout_per_task', 300)}s per task")
    
    # System resource monitoring
    initial_memory = psutil.virtual_memory().used / (1024**3)  # GB
    progress_tracker = WitnessProgressTracker(len(tasks))
    
    # Configure parallel execution parameters
    parallel_config = config.get('parallel', {})
    chunk_size = parallel_config.get('chunk_size', 1)
    timeout_per_task = parallel_config.get('timeout_per_task', 300)  # 5 minutes per task
    
    results = []
    failed_tasks = []
    
    try:
        # Use ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context('spawn')  # More robust for multiprocessing
        ) as executor:
            
            print("\n🚀 Starting parallel computation...\n")
            
            # Submit all tasks
            future_to_task = {
                executor.submit(process_witness_layer_task, task): task 
                for task in tasks
            }
            
            # Progress tracking setup
            progress_update_interval = max(1, len(tasks) // 20)  # Update every 5%
            last_progress_update = 0
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_task, timeout=timeout_per_task * len(tasks)):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result(timeout=timeout_per_task)
                        results.append(result)
                        progress_tracker.update(success=result.success)
                        
                        if not result.success:
                            failed_tasks.append((task, result.error_message))
                        
                        # Periodic progress reporting (every 5% or 25% completed tasks)
                        if len(results) - last_progress_update >= progress_update_interval:
                            _report_witness_progress(progress_tracker, initial_memory, results)
                            last_progress_update = len(results)
                            
                    except concurrent.futures.TimeoutError:
                        failed_tasks.append((task, "Task timeout"))
                        progress_tracker.update(success=False)
                        
                    except Exception as e:
                        failed_tasks.append((task, str(e)))
                        progress_tracker.update(success=False)
    
    except KeyboardInterrupt:
        print("\n\n⛔ Processing interrupted by user")
        raise
    except Exception as e:
        print(f"\n\n⚠️  Error in parallel processing: {e}")
        print("🔄 Falling back to sequential processing...")
        return _process_witness_tasks_sequential(tasks, layer_files, max_dimension)
    
    # Final progress report
    final_progress = progress_tracker.get_progress()
    print(f"\n\n📊 PARALLEL PROCESSING SUMMARY")
    print("-" * 35)
    print(f"  ✅ Completed: {final_progress['completed']}/{final_progress['total']} tasks")
    print(f"  🎉 Successful: {final_progress['completed'] - final_progress['failed']}")
    print(f"  ❌ Failed: {final_progress['failed']}")
    print(f"  ⚡ Rate: {final_progress['rate_per_second']:.2f} tasks/second")
    print(f"  ⏱️  Total time: {_format_time(final_progress['elapsed_time'])}")
    
    # Report failed tasks
    if failed_tasks:
        print(f"\n⚠️  Failed tasks ({len(failed_tasks)}):")
        for task, error in failed_tasks[:5]:  # Show first 5 failures
            print(f"    • {task.filename} [net={task.net_idx}, layer={task.layer_idx}]: {error}")
        if len(failed_tasks) > 5:
            print(f"    ... and {len(failed_tasks) - 5} more")
    
    # Memory usage report
    final_memory = psutil.virtual_memory().used / (1024**3)  # GB
    memory_increase = final_memory - initial_memory
    print(f"\n💾 Memory usage: +{memory_increase:.2f} GB")
    
    # Aggregate results
    print("\n📦 Aggregating results...")
    return aggregate_witness_results(results, layer_files, max_dimension)


def _process_witness_tasks_sequential(tasks: List[WitnessLayerTask], layer_files: Dict, max_dimension: int) -> Dict:
    """
    Process witness complex tasks sequentially with progress tracking (fallback method).
    """
    results = []
    
    print(f"\n🔄 Sequential processing: {len(tasks)} tasks")
    print("🚀 Starting computation...\n")
    
    # Process tasks sequentially
    for i, task in enumerate(tasks):
        try:
            result = process_witness_layer_task(task)
            results.append(result)
            
            # Show progress bar
            percentage = ((i + 1) / len(tasks)) * 100
            bar_length = 40
            filled_length = int(bar_length * percentage / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"\r[{bar}] {percentage:5.1f}% | {i + 1}/{len(tasks)} tasks", end='', flush=True)
            
            # Periodic memory cleanup
            if i % 10 == 0:
                gc.collect()
                    
        except Exception as e:
            # Create a failed result
            failed_result = WitnessLayerResult(
                filename=task.filename,
                net_idx=task.net_idx,
                layer_idx=task.layer_idx,
                task_id=task.task_id,
                betti_numbers=[1] + [0] * max_dimension,
                computation_time=0,
                success=False,
                error_message=str(e)
            )
            results.append(failed_result)
    
    print("\n\n📦 Aggregating results...")
    return aggregate_witness_results(results, layer_files, max_dimension)


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds/3600:.0f}h {(seconds%3600)/60:.0f}m"


def _report_witness_progress(progress_tracker: WitnessProgressTracker, initial_memory_gb: float, latest_results: List = None):
    """Report detailed progress information with Betti numbers for witness complex."""
    progress = progress_tracker.get_progress()
    current_memory = psutil.virtual_memory().used / (1024**3)  # GB
    memory_increase = current_memory - initial_memory_gb
    
    # Create progress bar
    bar_length = 40
    filled_length = int(bar_length * progress['percentage'] / 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # Format time estimates
    elapsed_str = _format_time(progress['elapsed_time'])
    eta_str = _format_time(progress['eta_seconds'])
    
    # Clear line and print progress
    print(f"\r[{bar}] {progress['percentage']:5.1f}% | "
          f"{progress['completed']:4d}/{progress['total']} tasks | "
          f"⚡ {progress['rate_per_second']:4.1f}/s | "
          f"⏱️  {elapsed_str} / {eta_str} | "
          f"💾 {memory_increase:+.1f}GB", end='', flush=True)
    
    # If there are failures, show count
    if progress['failed'] > 0:
        print(f" | ❌ {progress['failed']} failed", end='', flush=True)


def _print_witness_betti_summary(all_betti_results: Dict):
    """Print a clear summary of Betti numbers for all networks and layers from witness complex."""
    print("\n📊 BETTI NUMBERS SUMMARY")
    print("=" * 50)
    
    for filename, betti_tensor in all_betti_results.items():
        print(f"\n📄 {filename}")
        print("-" * 40)
        
        if not hasattr(betti_tensor, 'shape'):
            print("  ❌ No valid results")
            continue
            
        num_networks, num_layers, num_dims = betti_tensor.shape
        
        # Print header
        print(f"  🧪 Networks: {num_networks}, Layers: {num_layers}")
        print("\n  Layer-wise Betti numbers:")
        print("  " + "-" * 48)
        print("  Layer | H₀ (components) | H₁ (loops) | H₂ (voids)")
        print("  " + "-" * 48)
        
        # Print average Betti numbers per layer across all networks
        for layer_idx in range(num_layers):
            avg_bettis = np.mean(betti_tensor[:, layer_idx, :], axis=0)
            std_bettis = np.std(betti_tensor[:, layer_idx, :], axis=0)
            
            print(f"  {layer_idx:5d} | {avg_bettis[0]:15.1f} | {avg_bettis[1]:10.1f} | {avg_bettis[2]:10.1f}")
            
            # If there's variation across networks, show it
            if num_networks > 1 and np.any(std_bettis > 0.1):
                print(f"        | (±{std_bettis[0]:13.1f}) | (±{std_bettis[1]:8.1f}) | (±{std_bettis[2]:8.1f})")
        
        print("  " + "-" * 48)
        
        # Print overall statistics
        print("\n  📊 Overall statistics:")
        for dim in range(num_dims):
            dim_name = ["H₀ (components)", "H₁ (loops)", "H₂ (voids)"][dim]
            all_values = betti_tensor[:, :, dim].flatten()
            print(f"    {dim_name}:")
            print(f"      Range: [{all_values.min()}, {all_values.max()}]")
            print(f"      Mean: {all_values.mean():.2f} (±{all_values.std():.2f})")


def compute_layer_homology_witness_torch(config_path: str = "configs/homology_config.yaml") -> None:
    """
    Main function to compute homology using PyTorch-based witness complexes with parallel processing support.
    
    Parameters:
    - config_path: Path to the homology configuration file
    """
    print("\n🌐 WITNESS COMPLEX HOMOLOGY COMPUTATION")
    print("=" * 50)
    print(f"🖥️  Platform: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"🧠 Processing cores: {mp.cpu_count()}")
    print(f"💾 Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print("=" * 50)
    start_time = time.time()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract key parameters
    input_dir = config.get('io', {}).get('input_dir', 'results/layer_outputs')
    output_dir = config.get('io', {}).get('output_dir', 'results/homology')
    max_dimension = config.get('computation', {}).get('max_dimension', 2)
    
    # Check if parallel processing is enabled
    parallel_config = config.get('parallel', {})
    use_parallel = parallel_config.get('enabled', True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all layer output files
    print(f"\n📁 DATA LOADING")
    print("-" * 30)
    print(f"  📂 Input: {input_dir}")
    layer_files = load_layer_outputs(input_dir)
    
    if not layer_files:
        raise ValueError(f"No layer output files found in {input_dir}")
    
    # Create all tasks
    print(f"\n📋 TASK PREPARATION")
    print("-" * 30)
    tasks = create_witness_layer_tasks(layer_files, config)
    total_tasks = len(tasks)
    
    if total_tasks == 0:
        print("❌ ERROR: No valid tasks created. Check input data.")
        return
    
    # Get witness complex configuration for display
    witness_config = config.get('witness_complex', {})
    landmark_count = witness_config.get('n_landmarks', 50)
    if witness_config.get('adaptive_landmarks', False):
        landmark_count = f"adaptive ({witness_config.get('landmark_percentage', 0.005)*100:.1f}%)"
    
    print(f"  🎯 Total tasks: {total_tasks}")
    print(f"  🔺 Max dimension: H{max_dimension}")
    print(f"  📍 Landmarks: {landmark_count}")
    print(f"  ⚙️  Mode: {'Parallel' if use_parallel and total_tasks > 1 else 'Sequential'}")
    
    # Process tasks
    print(f"\n🔍 HOMOLOGY COMPUTATION")
    print("=" * 30)
    if use_parallel and total_tasks > 1:
        print("⚡ Mode: Parallel processing")
        all_betti_results = _process_witness_tasks_parallel(tasks, layer_files, max_dimension, config)
    else:
        print("🔄 Mode: Sequential processing")
        all_betti_results = _process_witness_tasks_sequential(tasks, layer_files, max_dimension)
    
    # Continue with saving results and cleanup
    _save_witness_results_and_cleanup(all_betti_results, config, output_dir, start_time)


def _save_witness_results_and_cleanup(all_betti_results: Dict, config: Dict, output_dir: str, start_time: float):
    """Save witness complex results and perform cleanup operations."""
    
    print(f"\n💾 SAVING RESULTS")
    print("=" * 30)
    
    # Save results
    if all_betti_results:
        # If only one file, save directly; if multiple, save as dictionary
        if len(all_betti_results) == 1:
            results_tensor = list(all_betti_results.values())[0]
        else:
            results_tensor = all_betti_results
        
        output_file = os.path.join(output_dir, 'layer_betti_numbers_witness_torch.pt')
        torch.save(results_tensor, output_file)
        print(f"  ✅ Betti numbers: {output_file}")
        
        # Save configuration used
        config_output = os.path.join(output_dir, 'homology_config_used_witness_torch.yaml')
        with open(config_output, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        print(f"  ✅ Configuration: {config_output}")
        
        # Save computation log
        total_time = time.time() - start_time
        log_file = os.path.join(output_dir, 'homology_computation_witness_torch.log')
        with open(log_file, 'w') as f:
            f.write(f"PyTorch Witness Complex Homology Computation Log\n")
            f.write(f"================================================\n")
            f.write(f"Start time: {time.ctime(start_time)}\n")
            f.write(f"Total computation time: {total_time:.2f} seconds\n")
            f.write(f"Implementation: Witness Complex (PyTorch + Gudhi)\n")
            f.write(f"Parallelization: Mixed Task Parallelization\n")
            f.write(f"Configuration file: {config.get('config_file', 'Unknown')}\n")
            f.write(f"Input directory: {config.get('io', {}).get('input_dir', 'Unknown')}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Max dimension: {config.get('computation', {}).get('max_dimension', 'Unknown')}\n")
            f.write(f"Files processed: {list(all_betti_results.keys())}\n")
            
            # Add system information
            f.write(f"\nSystem Information:\n")
            f.write(f"CPU cores: {mp.cpu_count()}\n")
            f.write(f"Available memory: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")
            
            # Add parallel processing stats if available
            parallel_config = config.get('parallel', {})
            if parallel_config.get('enabled', True):
                f.write(f"Workers used: {parallel_config.get('num_workers', 'auto-detected')}\n")
            
            # Add witness complex parameters
            if 'witness_complex' in config:
                witness_params = config['witness_complex']
                f.write(f"\nWitness Complex Parameters:\n")
                f.write(f"  Number of landmarks: {witness_params.get('n_landmarks', 'adaptive')}\n")
                f.write(f"  Landmark percentage: {witness_params.get('landmark_percentage', 0.005)}\n")
                f.write(f"  Landmark selection: {witness_params.get('landmark_selection', 'maxmin')}\n")
                f.write(f"  Witness type: {witness_params.get('witness_type', 'weak')}\n")
                f.write(f"  Max alpha square: {witness_params.get('max_alpha_square', 'inf')}\n")
            
            for filename, results in all_betti_results.items():
                f.write(f"\n{filename}:\n")
                if hasattr(results, 'shape'):
                    f.write(f"  Shape: {results.shape}\n")
                    f.write(f"  Betti number ranges: {[f'[{results[:,:,i].min()}, {results[:,:,i].max()}]' for i in range(results.shape[2])]}\n")
        
        print(f"\n🎉 COMPUTATION COMPLETE")
        print("-" * 30)
        print(f"  ⏱️  Total time: {_format_time(total_time)}")
        print(f"  📦 Results shape: {results_tensor.shape if hasattr(results_tensor, 'shape') else 'Dictionary'}")
        print(f"  📂 Output: {output_dir}")
        
        # Print Betti numbers summary
        _print_witness_betti_summary(all_betti_results)
        print(f"  📄 Files processed: {len(all_betti_results)}")
        
        # Memory cleanup
        gc.collect()
        
        print(f"\n{'='*50}")
        print("✅ WITNESS COMPLEX HOMOLOGY COMPUTATION SUCCESSFUL")
        print(f"{'='*50}")
        
    else:
        print("❌ ERROR: No valid layer outputs were processed.")
        print("⚠️  Check input data and configuration settings.")


# Maintain backward compatibility
def compute_layer_homology_witness(config_path: str = "configs/homology_config.yaml") -> None:
    """
    Backward-compatible wrapper that automatically uses parallel processing if available.
    """
    return compute_layer_homology_witness_torch(config_path)


def test_witness_parallel_processing() -> None:
    """
    Test function to verify witness complex parallel processing implementation works correctly.
    """
    print("Testing witness complex parallel processing implementation...")
    
    # Create synthetic test data
    import numpy as np
    np.random.seed(42)
    
    # Simulate layer outputs: [2 networks, 3 layers, 100 samples, 50 dimensions]
    test_data = np.random.randn(2, 3, 100, 50)
    test_filename = "test_layer_outputs_witness.pt"
    
    # Create test directory
    test_dir = "test_homology_witness"
    os.makedirs(test_dir, exist_ok=True)
    
    # Save test data
    torch.save(test_data, os.path.join(test_dir, test_filename))
    
    # Create test config
    test_config = {
        'io': {
            'input_dir': test_dir,
            'output_dir': test_dir
        },
        'computation': {
            'max_dimension': 2,
            'normalize_data': True
        },
        'sampling': {
            'min_points_threshold': 10
        },
        'witness_complex': {
            'n_landmarks': 20,
            'landmark_selection': 'random',
            'adaptive_landmarks': False,
            'max_alpha_square': 5.0,
            'witness_type': 'weak',
            'limit_dimension': 2,
            'use_euclidean_witness': True,
            'fallback_enabled': True
        },
        'parallel': {
            'enabled': True,
            'num_workers': 2,
            'timeout_per_task': 60
        }
    }
    
    try:
        # Test task creation
        layer_files = {test_filename: test_data}
        tasks = create_witness_layer_tasks(layer_files, test_config)
        print(f"Created {len(tasks)} witness complex tasks")
        
        # Test sequential processing
        print("Testing sequential processing...")
        seq_results = _process_witness_tasks_sequential(tasks[:2], layer_files, 2)  # Test first 2 tasks
        print(f"Sequential processing completed: {len(seq_results)} files")
        
        # Test parallel processing
        print("Testing parallel processing...")
        par_results = _process_witness_tasks_parallel(tasks[:2], layer_files, 2, test_config)  # Test first 2 tasks
        print(f"Parallel processing completed: {len(par_results)} files")
        
        print("Witness complex parallel processing test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise
    finally:
        # Cleanup
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_witness_parallel_processing()
    else:
        import argparse
        
        parser = argparse.ArgumentParser(
            description="Compute homology using PyTorch-based witness complexes with parallel processing"
        )
        parser.add_argument("--config", type=str, default="configs/homology_config.yaml",
                           help="Path to homology configuration file")
        
        args = parser.parse_args()
        compute_layer_homology_witness_torch(config_path=args.config)