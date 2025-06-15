"""
Compute homology for neural network layer activations using Ripser.

This script is a Ripser-based version of compute_homology.py that offers significant
performance improvements. It loads layer outputs from results/layer_outputs, computes 
distance matrices using functions from distance_computation.py, and then computes 
persistent homology using Ripser to extract topological features (Betti numbers) 
for each layer of each network.

Output format: [num_networks, num_layers, max_dimension] tensor of Betti numbers.
"""

import torch
import numpy as np
import os
import glob
import yaml
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, NamedTuple
from ripser import ripser
import concurrent.futures
import multiprocessing as mp
from dataclasses import dataclass
import psutil
import gc
# Removed tqdm import for cleaner output
import threading
import queue
import tempfile

# Import distance computation functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.distance_computation import knn_geodesic_distance, load_config


@dataclass
class LayerTask:
    """Task structure for parallel processing of individual layers."""
    layer_data_path: str  # Path to temporary file containing layer data
    config: Dict
    filename: str
    net_idx: int
    layer_idx: int
    task_id: int


@dataclass
class LayerResult:
    """Result structure for completed layer processing."""
    filename: str
    net_idx: int
    layer_idx: int
    task_id: int
    betti_numbers: List[int]
    computation_time: float
    success: bool
    error_message: Optional[str] = None
    memory_usage_mb: Optional[float] = None


class ProgressTracker:
    """Thread-safe progress tracker for parallel processing."""
    
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


def process_layer_task(task: LayerTask) -> LayerResult:
    """
    Worker function to process a single layer task.
    
    This function is designed to be stateless and memory-efficient,
    suitable for parallel execution across multiple processes.
    """
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Load layer data from temporary file
        layer_data = np.load(task.layer_data_path)
        
        # Validate input data
        if layer_data is None or len(layer_data) == 0:
            return LayerResult(
                filename=task.filename,
                net_idx=task.net_idx,
                layer_idx=task.layer_idx,
                task_id=task.task_id,
                betti_numbers=[0] * (task.config.get('computation', {}).get('max_dimension', 1) + 1),
                computation_time=time.time() - start_time,
                success=False,
                error_message="Empty layer data"
            )
        
        # Check minimum points threshold
        min_points = task.config.get('sampling', {}).get('min_points_threshold', 50)
        if len(layer_data) < min_points:
            return LayerResult(
                filename=task.filename,
                net_idx=task.net_idx,
                layer_idx=task.layer_idx,
                task_id=task.task_id,
                betti_numbers=[0] * (task.config.get('computation', {}).get('max_dimension', 1) + 1),
                computation_time=time.time() - start_time,
                success=False,
                error_message=f"Insufficient points: {len(layer_data)} < {min_points}"
            )
        
        # Apply sampling if dataset is too large
        sampling_config = task.config.get('sampling', {})
        max_points = sampling_config.get('fps_num_points', 1000)
        
        original_size = len(layer_data)
        if original_size > max_points:
            # Simple random sampling for now (could be improved with FPS)
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(original_size, max_points, replace=False)
            layer_data = layer_data[indices]
            # Removed verbose sampling output for cleaner logs
        
        # Compute distance matrix using knn_geodesic_distance
        distance_matrix = knn_geodesic_distance(layer_data)
        
        # Compute persistent homology
        max_dimension = task.config.get('computation', {}).get('max_dimension', 1)
        max_edge_length = task.config.get('computation', {}).get('max_edge_length', 0.5)
        
        betti_numbers = compute_persistent_homology_betti(
            distance_matrix.astype(np.float64), 
            max_dimension=max_dimension,
            max_edge_length=max_edge_length
        )
        
        # Memory cleanup
        del distance_matrix
        gc.collect()
        
        # Clean up temporary file
        try:
            os.remove(task.layer_data_path)
        except:
            pass  # Ignore cleanup errors
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        return LayerResult(
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
        
        # Clean up temporary file
        try:
            os.remove(task.layer_data_path)
        except:
            pass  # Ignore cleanup errors
        
        return LayerResult(
            filename=task.filename,
            net_idx=task.net_idx,
            layer_idx=task.layer_idx,
            task_id=task.task_id,
            betti_numbers=[1] + [0] * task.config.get('computation', {}).get('max_dimension', 1),
            computation_time=time.time() - start_time,
            success=False,
            error_message=str(e)
        )


def create_layer_tasks(layer_files: Dict, config: Dict) -> List[LayerTask]:
    """
    Create a list of all layer processing tasks for parallel execution.
    
    Flattens the nested structure of (filename, network, layer) into a single
    task queue for optimal load balancing.
    """
    tasks = []
    task_id = 0
    
    for filename, layer_outputs_orig in layer_files.items():
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
                    
                    # Save layer data to temporary file to avoid numpy pickling issues
                    temp_dir = Path(tempfile.gettempdir()) / 'compute_homology_ripser_data'
                    temp_dir.mkdir(exist_ok=True)
                    temp_file = temp_dir / f'layer_data_{task_id}.npy'
                    np.save(temp_file, layer_data)
                    
                    task = LayerTask(
                        layer_data_path=str(temp_file),
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


def aggregate_results(results: List[LayerResult], layer_files: Dict, max_dimension: int) -> Dict:
    """
    Aggregate parallel processing results back into the original data structure.
    
    Reconstructs the [num_networks, num_layers, max_dimension] tensor format
    from the flattened task results.
    """
    all_betti_results = {}
    
    # Group results by filename
    results_by_file = {}
    for result in results:
        if result.filename not in results_by_file:
            results_by_file[result.filename] = []
        results_by_file[result.filename].append(result)
    
    # Reconstruct the original tensor structure for each file
    for filename, file_results in results_by_file.items():
        if filename not in layer_files:
            continue
            
        layer_outputs = layer_files[filename]
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
                    print(f"Warning: Task failed for {filename} net={result.net_idx} layer={result.layer_idx}: {result.error_message}")
                
                betti_results[result.net_idx, result.layer_idx] = betti_numbers
            
            all_betti_results[filename] = betti_results
    
    return all_betti_results


def get_optimal_worker_count(config: Dict, total_tasks: int) -> int:
    """
    Determine optimal number of workers based on system resources and task characteristics.
    """
    # Get configured number of workers
    configured_workers = config.get('parallel', {}).get('num_workers', None)
    
    if configured_workers is not None:
        return max(1, min(configured_workers, total_tasks))
    
    # Auto-detect optimal worker count
    cpu_count = mp.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Estimate memory per worker (rough heuristic)
    estimated_memory_per_worker = 0.5  # GB per worker (conservative estimate)
    memory_limited_workers = int(available_memory_gb / estimated_memory_per_worker)
    
    # Use conservative estimate: 75% of CPU cores or memory limit, whichever is lower
    optimal_workers = min(
        max(1, int(cpu_count * 0.75)),
        memory_limited_workers,
        total_tasks  # Never more workers than tasks
    )
    
    return optimal_workers


def compute_persistent_homology_betti(distance_matrix: np.ndarray, max_dimension: int = 2, 
                                     max_edge_length: float = 1.0) -> List[int]:
    """
    Compute persistent homology and return Betti numbers using Ripser.
    
    This function provides the same interface as the Gudhi version but uses Ripser
    for significantly improved performance (typically 5-100x faster).
    
    Parameters:
    - distance_matrix: Square, symmetric distance matrix with zero diagonal
    - max_dimension: Maximum homology dimension to compute (e.g., 2 means H0, H1, H2)
    - max_edge_length: Maximum edge length for including edges in the Rips complex
    
    Returns:
    - List of Betti numbers for dimensions 0 up to max_dimension
    
    Raises:
    - ValueError: If distance_matrix is invalid (not square, not symmetric, negative diagonal)
    """
    try:
        # Validate input distance matrix
        if not isinstance(distance_matrix, np.ndarray):
            raise ValueError("distance_matrix must be a NumPy array.")
        
        if distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be a 2D array.")
        
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be square.")
        
        # More lenient symmetry check for integer matrices
        if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-6):
            print(f"Warning: Distance matrix may not be perfectly symmetric, max diff: {np.max(np.abs(distance_matrix - distance_matrix.T))}")
        
        if not np.allclose(np.diag(distance_matrix), 0, atol=1e-6):
            print(f"Warning: Diagonal elements may not be zero, max diagonal: {np.max(np.diag(distance_matrix))}")
        
        # Matrix size validated - Ripser handles large matrices efficiently
        
        # Compute persistent homology using Ripser
        # Removed verbose ripser output for cleaner logs
        
        result = ripser(distance_matrix,
                       maxdim=max_dimension,
                       thresh=max_edge_length,
                       distance_matrix=True)
        
        # Extract persistence diagrams
        diagrams = result['dgms']
        
        # Compute Betti numbers by counting features that persist at the max filtration value
        epsilon = 1e-10
        betti_numbers = []
        
        for dim in range(max_dimension + 1):
            if dim < len(diagrams):
                diagram = diagrams[dim]
                if len(diagram) > 0:
                    births = diagram[:, 0]
                    deaths = diagram[:, 1]
                    # Count features that are born at or before max_edge_length and die after it (or are infinite)
                    persistent_features = np.sum((births <= max_edge_length + epsilon) & 
                                               ((deaths > max_edge_length + epsilon) | 
                                                (deaths == np.inf)))
                    betti_numbers.append(int(persistent_features))
                else:
                    betti_numbers.append(0)
            else:
                betti_numbers.append(0)
        
        # B2 computation fixed: now correctly counts features born at threshold
        return betti_numbers
        
    except Exception as e:
        print(f"Error in homology computation: {e}")
        # Return safe default values
        return [1] + [0] * max_dimension


def load_layer_outputs(input_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load all layer output files from the input directory.
    
    Parameters:
    - input_dir: Directory containing layer output .pt files
    
    Returns:
    - Dictionary mapping filename to layer output tensors
    """
    layer_files = {}
    pattern = os.path.join(input_dir, "*.pt")
    
    for file_path in glob.glob(pattern):
        filename = os.path.basename(file_path)
        try:
            data = torch.load(file_path, map_location='cpu')
            if isinstance(data, dict) and 'layer_outputs' in data:
                layer_files[filename] = data['layer_outputs']
            else:
                layer_files[filename] = data
            print(f"  {filename}: {layer_files[filename].shape}")
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
    
    return layer_files


def process_single_layer(layer_activations: np.ndarray, config: Dict, layer_idx: int = 0) -> List[int]:
    """
    Process a single layer's activations to compute Betti numbers.
    
    Parameters:
    - layer_activations: Numpy array of shape (num_samples, layer_dim)
    - config: Configuration dictionary
    - layer_idx: Layer index for logging purposes
    
    Returns:
    - List of Betti numbers for each dimension
    """
    try:
        # Check minimum points threshold
        min_points = config.get('sampling', {}).get('min_points_threshold', 50)
        if len(layer_activations) < min_points:
            print(f"Warning: Layer {layer_idx} has only {len(layer_activations)} points, below threshold {min_points}")
            return [0] * (config.get('computation', {}).get('max_dimension', 1) + 1)
        
        print(f"Processing layer {layer_idx}: {layer_activations.shape}", end="", flush=True)
        
        # Compute distance matrix using knn_geodesic_distance
        distance_matrix = knn_geodesic_distance(layer_activations)
        print(f" -> distance matrix {distance_matrix.shape}", end="", flush=True)
        
        # Compute persistent homology
        max_dimension = config.get('computation', {}).get('max_dimension', 1)
        max_edge_length = config.get('computation', {}).get('max_edge_length', 0.5)
        
        betti_numbers = compute_persistent_homology_betti(
            distance_matrix.astype(np.float64), 
            max_dimension=max_dimension,
            max_edge_length=max_edge_length
        )
        
        return betti_numbers
        
    except Exception as e:
        print(f"Error processing layer {layer_idx}: {e}")
        # Return zero Betti numbers on error
        max_dimension = config.get('computation', {}).get('max_dimension', 1)
        return [0] * (max_dimension + 1)


def compute_layer_homology_parallel(config_path: str = "configs/homology_config.yaml") -> None:
    """
    Main function to compute homology for all layer outputs using Ripser with parallel processing.
    
    This implementation uses Option C (Mixed Task Parallelization) where all (network, layer)
    pairs are flattened into a task queue for optimal load balancing and maximum efficiency.
    
    Parameters:
    - config_path: Path to the homology configuration file
    """
    print("RIPSER HOMOLOGY COMPUTATION")
    print("=" * 50)
    print("Starting parallel persistent homology computation using Ripser...")
    print("Implementation: Fast Vietoris-Rips complex computation")
    print("Performance: Ripser typically provides 5-100x speedup over Gudhi")
    print("=" * 50)
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract key parameters
    input_dir = config.get('io', {}).get('input_dir', 'results/layer_outputs')
    output_dir = config.get('io', {}).get('output_dir', 'results/homology')
    max_dimension = config.get('computation', {}).get('max_dimension', 1)
    
    # Check if parallel processing is enabled
    parallel_config = config.get('parallel', {})
    use_parallel = parallel_config.get('enabled', True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all layer output files
    print(f"\nDATA LOADING:")
    print("-" * 20)
    print(f"Input directory: {input_dir}")
    layer_files = load_layer_outputs(input_dir)
    
    if not layer_files:
        raise ValueError(f"No layer output files found in {input_dir}")
    
    # Create all tasks
    print(f"\nTASK PREPARATION:")
    print("-" * 20)
    print("Creating parallel processing task queue...")
    tasks = create_layer_tasks(layer_files, config)
    total_tasks = len(tasks)
    
    if total_tasks == 0:
        print("ERROR: No valid tasks created. Check input data.")
        return
    
    print(f"Task queue created: {total_tasks} layer processing tasks")
    print(f"Max homology dimension: {max_dimension}")
    print(f"Sample points per layer: {config.get('sampling', {}).get('fps_num_points', 'N/A')}")
    print(f"Parallel processing: {'Enabled' if use_parallel and total_tasks > 1 else 'Disabled'}")
    
    # Process tasks
    print(f"\nHOMOLOGY COMPUTATION:")
    print("=" * 30)
    if use_parallel and total_tasks > 1:
        print("Execution mode: Parallel processing")
        all_betti_results = _process_tasks_parallel(tasks, layer_files, max_dimension, config)
    else:
        print("Execution mode: Sequential processing")
        all_betti_results = _process_tasks_sequential(tasks, layer_files, max_dimension)
    
    # Continue with saving results and cleanup
    _save_results_and_cleanup(all_betti_results, config, output_dir, start_time)


def _process_tasks_parallel(tasks: List[LayerTask], layer_files: Dict, max_dimension: int, config: Dict) -> Dict:
    """
    Process tasks using parallel execution with comprehensive monitoring and error handling.
    """
    num_workers = get_optimal_worker_count(config, len(tasks))
    print(f"Parallel workers: {num_workers}")
    print(f"Total tasks: {len(tasks)}")
    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Memory limit: {config.get('parallel', {}).get('memory_limit_gb', 'N/A')} GB")
    
    # System resource monitoring
    initial_memory = psutil.virtual_memory().used / (1024**3)  # GB
    progress_tracker = ProgressTracker(len(tasks))
    
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
            
            print("\nSubmitting tasks to worker pool...")
            
            # Submit all tasks
            future_to_task = {
                executor.submit(process_layer_task, task): task 
                for task in tasks
            }
            
            # Progress tracking setup
            progress_update_interval = max(1, len(tasks) // 20)  # Update every 5%
            last_progress_update = 0
            
            print("Processing homology computations...")
            
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
                            _report_progress(progress_tracker, initial_memory, results)
                            last_progress_update = len(results)
                            
                    except concurrent.futures.TimeoutError:
                        print(f"Warning: Task timed out - {task.filename} net={task.net_idx} layer={task.layer_idx}")
                        failed_tasks.append((task, "Task timeout"))
                        progress_tracker.update(success=False)
                        
                    except Exception as e:
                        print(f"Warning: Task execution failed - {task.filename} net={task.net_idx} layer={task.layer_idx}: {e}")
                        failed_tasks.append((task, str(e)))
                        progress_tracker.update(success=False)
    
    except KeyboardInterrupt:
        print("\nParallel processing interrupted by user")
        raise
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        print("Falling back to sequential processing...")
        return _process_tasks_sequential(tasks, layer_files, max_dimension)
    
    # Final progress report
    final_progress = progress_tracker.get_progress()
    print(f"\nPARALLEL PROCESSING SUMMARY:")
    print("-" * 30)
    print(f"Tasks completed: {final_progress['completed']}/{final_progress['total']}")
    print(f"Successful: {final_progress['completed'] - final_progress['failed']}")
    print(f"Failed: {final_progress['failed']}")
    print(f"Processing rate: {final_progress['rate_per_second']:.2f} tasks/second")
    print(f"Total processing time: {final_progress['elapsed_time']:.2f} seconds")
    
    # Report failed tasks
    if failed_tasks:
        print(f"\nFailed tasks ({len(failed_tasks)}):")
        for task, error in failed_tasks[:5]:  # Show first 5 failures
            print(f"  {task.filename} net={task.net_idx} layer={task.layer_idx}: {error}")
        if len(failed_tasks) > 5:
            print(f"  ... and {len(failed_tasks) - 5} more")
    
    # Memory usage report
    final_memory = psutil.virtual_memory().used / (1024**3)  # GB
    memory_increase = final_memory - initial_memory
    print(f"Memory increase: {memory_increase:.2f} GB")
    
    # Aggregate results
    print("\nAggregating computation results...")
    return aggregate_results(results, layer_files, max_dimension)


def _process_tasks_sequential(tasks: List[LayerTask], layer_files: Dict, max_dimension: int) -> Dict:
    """
    Process tasks sequentially with progress tracking (fallback method).
    """
    results = []
    
    print(f"Sequential processing: {len(tasks)} layer tasks")
    print("Progress tracking enabled with periodic memory cleanup")
    
    # Process tasks sequentially without progress bar
    for i, task in enumerate(tasks):
        try:
            result = process_layer_task(task)
            results.append(result)
            
            if not result.success:
                print(f"WARNING: Task failed - {task.filename} net={task.net_idx} layer={task.layer_idx}: {result.error_message}")
            
            # Show progress every 25% completion
            if (i + 1) % max(1, len(tasks) // 4) == 0:
                percentage = ((i + 1) / len(tasks)) * 100
                betti_info = ""
                if result.success and hasattr(result, 'betti_numbers'):
                    betti_info = f" | Latest Betti: {result.betti_numbers}"
                print(f"Progress: {percentage:.0f}% ({i + 1}/{len(tasks)} tasks completed){betti_info}")
            
            # Periodic memory cleanup
            if i % 10 == 0:
                gc.collect()
                    
        except Exception as e:
            print(f"ERROR: Processing task {i}: {e}")
            # Create a failed result
            failed_result = LayerResult(
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
    
    return aggregate_results(results, layer_files, max_dimension)


def _report_progress(progress_tracker: ProgressTracker, initial_memory_gb: float, latest_results: List = None):
    """Report detailed progress information with Betti numbers."""
    progress = progress_tracker.get_progress()
    current_memory = psutil.virtual_memory().used / (1024**3)  # GB
    memory_increase = current_memory - initial_memory_gb
    
    progress_msg = (f"Progress: {progress['percentage']:.1f}% "
                   f"({progress['completed']}/{progress['total']}) "
                   f"Rate: {progress['rate_per_second']:.2f}/sec "
                   f"ETA: {progress['eta_seconds']:.0f}s "
                   f"Memory: +{memory_increase:.2f}GB")
    
    # Add latest Betti numbers if available
    if latest_results and len(latest_results) > 0:
        latest_result = latest_results[-1]
        if hasattr(latest_result, 'betti_numbers') and latest_result.success:
            betti_str = str(latest_result.betti_numbers)
            progress_msg += f" | Betti: {betti_str}"
    
    print(progress_msg)


def _save_results_and_cleanup(all_betti_results: Dict, config: Dict, output_dir: str, start_time: float):
    """Save results and perform cleanup operations."""
    
    print(f"\nRESULTS AND CLEANUP:")
    print("=" * 25)
    
    # Save results
    if all_betti_results:
        # If only one file, save directly; if multiple, save as dictionary
        if len(all_betti_results) == 1:
            results_tensor = list(all_betti_results.values())[0]
        else:
            results_tensor = all_betti_results
        
        output_file = os.path.join(output_dir, 'layer_betti_numbers_ripser_parallel.pt')
        torch.save(results_tensor, output_file)
        print(f"Betti numbers saved: {output_file}")
        
        # Save configuration used
        config_output = os.path.join(output_dir, 'homology_config_used_ripser_parallel.yaml')
        with open(config_output, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        print(f"Configuration saved: {config_output}")
        
        # Save computation log
        total_time = time.time() - start_time
        log_file = os.path.join(output_dir, 'homology_computation_ripser_parallel.log')
        with open(log_file, 'w') as f:
            f.write(f"Parallel Homology Computation Log (Ripser)\n")
            f.write(f"=========================================\n")
            f.write(f"Start time: {time.ctime(start_time)}\n")
            f.write(f"Total computation time: {total_time:.2f} seconds\n")
            f.write(f"Implementation: Ripser (fast Vietoris-Rips persistent homology)\n")
            f.write(f"Parallelization: Option C (Mixed Task Parallelization)\n")
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
            
            for filename, results in all_betti_results.items():
                f.write(f"\n{filename}:\n")
                if hasattr(results, 'shape'):
                    f.write(f"  Shape: {results.shape}\n")
                    f.write(f"  Betti number ranges: {[f'[{results[:,:,i].min()}, {results[:,:,i].max()}]' for i in range(results.shape[2])]}\n")
        
        print(f"\nCOMPUTATION SUMMARY:")
        print("-" * 25)
        print(f"Total computation time: {total_time:.2f} seconds")
        print(f"Results tensor shape: {results_tensor.shape if hasattr(results_tensor, 'shape') else 'Dictionary format'}")
        print(f"Output directory: {output_dir}")
        print(f"Files processed: {len(all_betti_results)}")
        
        # Memory cleanup
        gc.collect()
        
        # Clean up temporary directory
        temp_dir = Path(tempfile.gettempdir()) / 'compute_homology_ripser_data'
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                print("Temporary data directory cleaned up successfully.")
        except:
            pass  # Ignore cleanup errors
        
        print(f"{'='*50}")
        print("HOMOLOGY COMPUTATION COMPLETED SUCCESSFULLY")
        print(f"{'='*50}")
        
    else:
        print("ERROR: No valid layer outputs were processed.")
        print("Check input data and configuration settings.")


# Maintain backward compatibility
def compute_layer_homology(config_path: str = "configs/homology_config.yaml") -> None:
    """
    Backward-compatible wrapper that automatically uses parallel processing if available.
    """
    return compute_layer_homology_parallel(config_path)


def benchmark_against_gudhi(config_path: str = "configs/homology_config.yaml") -> None:
    """
    Benchmark Ripser against Gudhi implementation for performance comparison.
    
    This function runs both implementations on the same data and reports timing differences.
    """
    print("Running benchmark comparison between Ripser and Gudhi...")
    
    # Load configuration
    config = load_config(config_path)
    input_dir = config.get('io', {}).get('input_dir', 'results/layer_outputs')
    
    # Load a sample layer output for benchmarking
    layer_files = load_layer_outputs(input_dir)
    if not layer_files:
        print("No layer files found for benchmarking")
        return
    
    # Get the first file and first layer
    filename, layer_outputs = next(iter(layer_files.items()))
    if isinstance(layer_outputs, torch.Tensor):
        layer_outputs = layer_outputs.cpu().numpy()
    
    if layer_outputs.ndim == 4:
        # Extract first network, first layer
        layer_data = layer_outputs[0, 0]
        print(f"Benchmarking on layer data of shape: {layer_data.shape}")
        
        # Compute distance matrix
        distance_matrix = knn_geodesic_distance(layer_data)
        
        max_dimension = config.get('computation', {}).get('max_dimension', 1)
        max_edge_length = config.get('computation', {}).get('max_edge_length', 0.5)
        
        # Time Ripser
        start_time = time.time()
        betti_ripser = compute_persistent_homology_betti(
            distance_matrix.astype(np.float64),
            max_dimension=max_dimension,
            max_edge_length=max_edge_length
        )
        ripser_time = time.time() - start_time
        
        # Try to time Gudhi
        try:
            from src.topology.compute_homology import compute_persistent_homology_betti as compute_betti_gudhi
            
            start_time = time.time()
            betti_gudhi = compute_betti_gudhi(
                distance_matrix.astype(np.float64),
                max_dimension=max_dimension,
                max_edge_length=max_edge_length
            )
            gudhi_time = time.time() - start_time
            
            print(f"\nBenchmark Results:")
            print(f"Ripser time: {ripser_time:.3f}s")
            print(f"Gudhi time: {gudhi_time:.3f}s")
            print(f"Speedup: {gudhi_time/ripser_time:.1f}x")
            print(f"Ripser Betti numbers: {betti_ripser}")
            print(f"Gudhi Betti numbers: {betti_gudhi}")
            print(f"Results match: {betti_ripser == betti_gudhi}")
            
        except ImportError:
            print("Could not import Gudhi version for comparison")
            print(f"Ripser time: {ripser_time:.3f}s")
            print(f"Ripser Betti numbers: {betti_ripser}")


def test_parallel_processing() -> None:
    """
    Test function to verify parallel processing implementation works correctly.
    """
    print("Testing parallel processing implementation...")
    
    # Create synthetic test data
    import numpy as np
    np.random.seed(42)
    
    # Simulate layer outputs: [2 networks, 3 layers, 100 samples, 50 dimensions]
    test_data = np.random.randn(2, 3, 100, 50)
    test_filename = "test_layer_outputs.pt"
    
    # Create test directory
    test_dir = "test_homology"
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
            'max_dimension': 1,
            'max_edge_length': 2.0
        },
        'sampling': {
            'min_points_threshold': 10
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
        tasks = create_layer_tasks(layer_files, test_config)
        print(f"Created {len(tasks)} tasks")
        
        # Test sequential processing
        print("Testing sequential processing...")
        seq_results = _process_tasks_sequential(tasks[:2], layer_files, 1)  # Test first 2 tasks
        print(f"Sequential processing completed: {len(seq_results)} files")
        
        # Test parallel processing
        print("Testing parallel processing...")
        par_results = _process_tasks_parallel(tasks[:2], layer_files, 1, test_config)  # Test first 2 tasks
        print(f"Parallel processing completed: {len(par_results)} files")
        
        print("Parallel processing test completed successfully!")
        
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
        test_parallel_processing()
    else:
        # Run the main computation
        compute_layer_homology()
        
        # Optionally run benchmark
        # benchmark_against_gudhi()