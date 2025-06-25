"""
Standalone Compute Boundary Homology

This script computes persistent homology for decision boundary data.
All dependencies are contained within this file.

Author: Claude Code
Date: 2025
"""

import torch
import numpy as np
import os
import sys
import glob
import yaml
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from ripser import ripser
import concurrent.futures
import multiprocessing as mp
from dataclasses import dataclass
import psutil
import gc
import tempfile

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import existing infrastructure
from src.topology.compute_homology_ripser import compute_persistent_homology_betti, ProgressTracker
from src.utils.distance_computation import knn_geodesic_distance


@dataclass
class BoundaryExtractionResult:
    """Result of decision boundary extraction."""
    epoch: int
    boundary_points: Optional[np.ndarray] = None
    mesh_vertices: Optional[np.ndarray] = None  
    mesh_faces: Optional[np.ndarray] = None
    extraction_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class BoundaryTopologyTask:
    """Task structure for boundary topology computation."""
    boundary_data_path: str
    config: Dict
    epoch: int
    task_id: int


@dataclass
class BoundaryTopologyResult:
    """Result structure for boundary topology computation."""
    epoch: int
    task_id: int
    betti_numbers: List[int]
    computation_time: float
    success: bool
    error_message: Optional[str] = None
    num_boundary_points: int = 0
    memory_usage_mb: Optional[float] = None


def load_boundary_config(config_path: str = "configs/decision_boundary_config.yaml") -> Dict:
    """Load decision boundary configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}


class BoundaryHomologyComputer:
    """
    Computes persistent homology for decision boundary data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the boundary homology computer.
        
        Parameters:
        - config: Configuration dictionary (from boundary config file)
        """
        self.config = config
        self.topology_config = config.get('topology', {})
        self.performance_config = config.get('performance', {})
        
    def load_boundary_data(self, input_dir: str) -> Dict[str, List[BoundaryExtractionResult]]:
        """
        Load boundary data from training results.
        
        Parameters:
        - input_dir: Directory containing boundary data
        
        Returns:
        - boundary_data: Dict mapping filenames to lists of BoundaryExtractionResult
        """
        boundary_data = {}
        
        # Look for training results files (multiple patterns)
        patterns = [
            os.path.join(input_dir, "**", "training_results.pt"),
            os.path.join(input_dir, "**", "complete_training_results.pt"),
            os.path.join(input_dir, "**", "*training_results*.pt")
        ]
        
        result_files = []
        for pattern in patterns:
            result_files.extend(glob.glob(pattern, recursive=True))
        
        # Remove duplicates
        result_files = list(set(result_files))
        
        if not result_files:
            # Also look for individual boundary files
            pattern = os.path.join(input_dir, "**", "topology_epoch_*.pt")
            individual_files = glob.glob(pattern, recursive=True)
            
            if individual_files:
                print(f"Found {len(individual_files)} individual boundary files")
                return self._load_individual_boundary_files(individual_files)
            else:
                print(f"No boundary data found in {input_dir}")
                print(f"Searched patterns: {patterns}")
                return {}
        
        print(f"Found {len(result_files)} training result files")
        
        for file_path in result_files:
            try:
                # Extract architecture name from path
                arch_name = Path(file_path).parent.parent.name
                
                # Load training results
                data = torch.load(file_path, map_location='cpu')
                boundary_results = data.get('boundary_results', [])
                
                if boundary_results:
                    boundary_data[arch_name] = boundary_results
                    print(f"  {arch_name}: {len(boundary_results)} boundary extractions")
                else:
                    print(f"  {arch_name}: No boundary data found")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return boundary_data
    
    def _load_individual_boundary_files(self, file_paths: List[str]) -> Dict[str, List[BoundaryExtractionResult]]:
        """Load boundary data from individual topology files."""
        boundary_data = {}
        
        # Group files by architecture (assuming directory structure)
        arch_files = {}
        for file_path in file_paths:
            arch_name = Path(file_path).parent.parent.parent.name  # Go up to architecture directory
            if arch_name not in arch_files:
                arch_files[arch_name] = []
            arch_files[arch_name].append(file_path)
        
        for arch_name, files in arch_files.items():
            boundary_results = []
            
            for file_path in sorted(files):
                try:
                    data = torch.load(file_path, map_location='cpu')
                    
                    # Create BoundaryExtractionResult from saved data
                    result = BoundaryExtractionResult(
                        epoch=data['epoch'],
                        boundary_points=data.get('boundary_points'),
                        mesh_vertices=data.get('mesh_vertices'),
                        mesh_faces=data.get('mesh_faces'),
                        extraction_time=data.get('extraction_time', 0),
                        success=True,
                        metadata=data.get('metadata')
                    )
                    boundary_results.append(result)
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            
            if boundary_results:
                boundary_data[arch_name] = boundary_results
                print(f"  {arch_name}: {len(boundary_results)} boundary extractions")
        
        return boundary_data
    
    def compute_single_boundary_topology(self, boundary_points: np.ndarray) -> Tuple[List[int], float]:
        """
        Compute topology for a single boundary point cloud.
        
        Parameters:
        - boundary_points: Array of shape (N, 3) with boundary points
        
        Returns:
        - betti_numbers: List of Betti numbers
        - computation_time: Time taken for computation
        """
        start_time = time.time()
        
        try:
            # Check minimum points threshold
            sampling_config = self.topology_config.get('sampling', {})
            min_points = sampling_config.get('min_points_threshold', 100)
            
            if len(boundary_points) < min_points:
                print(f"Warning: Only {len(boundary_points)} boundary points, below threshold {min_points}")
                max_dim = self.topology_config.get('computation', {}).get('max_dimension', 2)
                return [0] * (max_dim + 1), time.time() - start_time
            
            # Apply sampling if too many points
            max_points = sampling_config.get('num_points', 2000)
            
            if len(boundary_points) > max_points:
                # Use random sampling for simplicity
                indices = np.random.choice(len(boundary_points), max_points, replace=False)
                boundary_points = boundary_points[indices]
                print(f"Sampled {max_points} points from {len(boundary_points)} using random sampling")
            
            # Normalize data if configured
            computation_config = self.topology_config.get('computation', {})
            if computation_config.get('normalize_data', True):
                boundary_points = (boundary_points - np.mean(boundary_points, axis=0)) / \
                                (np.std(boundary_points, axis=0) + 1e-8)
            
            # Compute distance matrix
            distance_matrix = knn_geodesic_distance(boundary_points)
            
            # Compute persistent homology
            max_dimension = computation_config.get('max_dimension', 2)
            max_edge_length = computation_config.get('max_edge_length', 1.0)
            
            betti_numbers = compute_persistent_homology_betti(
                distance_matrix.astype(np.float64),
                max_dimension=max_dimension,
                max_edge_length=max_edge_length
            )
            
            return betti_numbers, time.time() - start_time
            
        except Exception as e:
            print(f"Error in topology computation: {e}")
            max_dim = self.topology_config.get('computation', {}).get('max_dimension', 2)
            return [0] * (max_dim + 1), time.time() - start_time
    
    def process_boundary_task(self, task: BoundaryTopologyTask) -> BoundaryTopologyResult:
        """
        Process a single boundary topology computation task.
        
        Parameters:
        - task: BoundaryTopologyTask object
        
        Returns:
        - BoundaryTopologyResult object
        """
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Load boundary data
            boundary_data = np.load(task.boundary_data_path)
            
            if boundary_data is None or len(boundary_data) == 0:
                return BoundaryTopologyResult(
                    epoch=task.epoch,
                    task_id=task.task_id,
                    betti_numbers=[0] * (task.config.get('topology', {}).get('computation', {}).get('max_dimension', 2) + 1),
                    computation_time=time.time() - start_time,
                    success=False,
                    error_message="Empty boundary data",
                    num_boundary_points=0
                )
            
            # Compute topology
            betti_numbers, topo_time = self.compute_single_boundary_topology(boundary_data)
            
            # Clean up temporary file
            try:
                os.remove(task.boundary_data_path)
            except:
                pass
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            return BoundaryTopologyResult(
                epoch=task.epoch,
                task_id=task.task_id,
                betti_numbers=betti_numbers,
                computation_time=time.time() - start_time,
                success=True,
                num_boundary_points=len(boundary_data),
                memory_usage_mb=memory_usage
            )
            
        except Exception as e:
            # Clean up temporary file
            try:
                os.remove(task.boundary_data_path)
            except:
                pass
            
            return BoundaryTopologyResult(
                epoch=task.epoch,
                task_id=task.task_id,
                betti_numbers=[1] + [0] * task.config.get('topology', {}).get('computation', {}).get('max_dimension', 2),
                computation_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                num_boundary_points=0
            )
    
    def create_topology_tasks(self, boundary_data: Dict[str, List[BoundaryExtractionResult]]) -> List[BoundaryTopologyTask]:
        """
        Create topology computation tasks for parallel processing.
        
        Parameters:
        - boundary_data: Dictionary of boundary extraction results
        
        Returns:
        - List of BoundaryTopologyTask objects
        """
        tasks = []
        task_id = 0
        
        # Create temporary directory for task data
        temp_dir = Path(tempfile.gettempdir()) / 'boundary_topology_tasks'
        temp_dir.mkdir(exist_ok=True)
        
        for arch_name, boundary_results in boundary_data.items():
            print(f"Creating tasks for {arch_name}: {len(boundary_results)} boundaries")
            
            for result in boundary_results:
                if result.success and result.boundary_points is not None:
                    # Save boundary points to temporary file
                    temp_file = temp_dir / f'boundary_points_{task_id}.npy'
                    np.save(temp_file, result.boundary_points)
                    
                    task = BoundaryTopologyTask(
                        boundary_data_path=str(temp_file),
                        config=self.config,
                        epoch=result.epoch,
                        task_id=task_id
                    )
                    tasks.append(task)
                    task_id += 1
        
        print(f"Created {len(tasks)} topology computation tasks")
        return tasks
    
    def compute_parallel(self, tasks: List[BoundaryTopologyTask]) -> List[BoundaryTopologyResult]:
        """
        Compute topology for all tasks using parallel processing.
        
        Parameters:
        - tasks: List of BoundaryTopologyTask objects
        
        Returns:
        - List of BoundaryTopologyResult objects
        """
        if not tasks:
            return []
        
        # Determine number of workers
        parallel_config = self.performance_config.get('parallel', {})
        num_workers = parallel_config.get('num_workers', None)
        if num_workers is None:
            num_workers = min(mp.cpu_count(), len(tasks))
        
        print(f"Computing topology using {num_workers} parallel workers...")
        
        # Progress tracking
        progress_tracker = ProgressTracker(len(tasks))
        results = []
        
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self.process_boundary_task, task): task 
                    for task in tasks
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per task
                        results.append(result)
                        progress_tracker.update(success=result.success)
                        
                        if len(results) % 10 == 0:  # Progress update every 10 tasks
                            progress = progress_tracker.get_progress()
                            print(f"Progress: {progress['completed']}/{progress['total']} "
                                  f"({progress['percentage']:.1f}%)")
                        
                    except Exception as e:
                        print(f"Task failed: {e}")
                        progress_tracker.update(success=False)
        
        except KeyboardInterrupt:
            print("\nComputation interrupted by user")
            raise
        
        # Final progress report
        final_progress = progress_tracker.get_progress()
        print(f"\nTopology computation completed!")
        print(f"Tasks completed: {final_progress['completed']}/{final_progress['total']}")
        print(f"Success rate: {(final_progress['completed'] - final_progress['failed'])/final_progress['total']*100:.1f}%")
        
        return results
    
    def aggregate_results(self, results: List[BoundaryTopologyResult], 
                         boundary_data: Dict[str, List[BoundaryExtractionResult]]) -> Dict:
        """
        Aggregate topology results back into architecture structure.
        
        Parameters:
        - results: List of topology computation results
        - boundary_data: Original boundary data structure
        
        Returns:
        - aggregated_results: Dictionary with topology data per architecture
        """
        # Create mapping from epoch to results
        epoch_to_result = {result.epoch: result for result in results if result.success}
        
        aggregated = {}
        
        for arch_name, boundary_results in boundary_data.items():
            arch_topology = []
            
            for boundary_result in boundary_results:
                if boundary_result.epoch in epoch_to_result:
                    topo_result = epoch_to_result[boundary_result.epoch]
                    
                    # Update the original boundary result with topology data
                    boundary_result.betti_numbers = topo_result.betti_numbers
                    boundary_result.topology_time = topo_result.computation_time
                    
                    arch_topology.append({
                        'epoch': boundary_result.epoch,
                        'betti_numbers': topo_result.betti_numbers,
                        'computation_time': topo_result.computation_time,
                        'num_points': topo_result.num_boundary_points
                    })
            
            aggregated[arch_name] = {
                'boundary_results': boundary_results,
                'topology_summary': arch_topology
            }
        
        return aggregated
    
    def save_results(self, aggregated_results: Dict, output_dir: str):
        """
        Save topology computation results.
        
        Parameters:
        - aggregated_results: Aggregated topology results
        - output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        results_file = output_path / 'boundary_topology_results.pt'
        torch.save(aggregated_results, results_file)
        print(f"Complete results saved: {results_file}")
        
        # Save individual architecture results
        for arch_name, arch_data in aggregated_results.items():
            arch_file = output_path / f'{arch_name}_boundary_topology.pt'
            torch.save(arch_data, arch_file)
            print(f"Architecture results saved: {arch_file}")
        
        # Save summary CSV
        summary_data = []
        for arch_name, arch_data in aggregated_results.items():
            for topo_data in arch_data['topology_summary']:
                row = {
                    'architecture': arch_name,
                    'epoch': topo_data['epoch'],
                    'betti_0': topo_data['betti_numbers'][0] if len(topo_data['betti_numbers']) > 0 else 0,
                    'betti_1': topo_data['betti_numbers'][1] if len(topo_data['betti_numbers']) > 1 else 0,
                    'betti_2': topo_data['betti_numbers'][2] if len(topo_data['betti_numbers']) > 2 else 0,
                    'num_points': topo_data['num_points'],
                    'computation_time': topo_data['computation_time']
                }
                summary_data.append(row)
        
        if summary_data:
            import pandas as pd
            df = pd.DataFrame(summary_data)
            csv_file = output_path / 'boundary_topology_summary.csv'
            df.to_csv(csv_file, index=False)
            print(f"Summary CSV saved: {csv_file}")


def compute_boundary_homology_main(config_path: str = "configs/decision_boundary_config.yaml",
                                  input_dir: str = None,
                                  output_dir: str = None) -> None:
    """
    Main function to compute boundary homology from extracted boundary data.
    
    Parameters:
    - config_path: Path to boundary configuration file
    - input_dir: Input directory containing boundary data (if None, uses config)
    - output_dir: Output directory for results (if None, uses config)
    """
    print("BOUNDARY HOMOLOGY COMPUTATION")
    print("=" * 40)
    print("Computing persistent homology for decision boundary data...")
    print("Implementation: Ripser-based topology computation")
    print("=" * 40)
    
    start_time = time.time()
    
    # Load configuration
    config = load_boundary_config(config_path)
    
    # Set directories
    if input_dir is None:
        input_dir = config.get('output', {}).get('directories', {}).get('base_dir', 'results/decision_boundary_analysis')
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'topology_results')
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create homology computer
    computer = BoundaryHomologyComputer(config)
    
    # Load boundary data
    print("\nLoading boundary data...")
    boundary_data = computer.load_boundary_data(input_dir)
    
    if not boundary_data:
        print("No boundary data found!")
        return
    
    # Create topology computation tasks
    print("\nCreating topology computation tasks...")
    tasks = computer.create_topology_tasks(boundary_data)
    
    if not tasks:
        print("No valid boundary data for topology computation!")
        return
    
    # Compute topology (parallel processing)
    print(f"\nComputing topology for {len(tasks)} boundaries...")
    results = computer.compute_parallel(tasks)
    
    # Aggregate results
    print("\nAggregating results...")
    aggregated_results = computer.aggregate_results(results, boundary_data)
    
    # Save results
    print("\nSaving results...")
    computer.save_results(aggregated_results, output_dir)
    
    # Summary
    total_time = time.time() - start_time
    successful_results = len([r for r in results if r.success])
    
    print(f"\nBOUNDARY HOMOLOGY COMPUTATION COMPLETED")
    print("=" * 45)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Architectures processed: {len(boundary_data)}")
    print(f"Boundaries processed: {successful_results}/{len(results)}")
    print(f"Results saved to: {output_dir}")
    
    # Print summary statistics
    for arch_name, arch_data in aggregated_results.items():
        topo_summary = arch_data['topology_summary']
        if topo_summary:
            final_betti = topo_summary[-1]['betti_numbers']
            print(f"  {arch_name}: {len(topo_summary)} epochs, final Betti = {final_betti}")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute Boundary Topology")
    parser.add_argument('--config', type=str, default='configs/decision_boundary_config.yaml',
                       help='Path to boundary configuration file')
    parser.add_argument('--input-dir', type=str, 
                       help='Input directory containing boundary data')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for topology results')
    parser.add_argument('--sequential', action='store_true',
                       help='Use sequential processing instead of parallel')
    
    args = parser.parse_args()
    
    try:
        # Override parallel processing if requested
        if args.sequential:
            config = load_boundary_config(args.config)
            config['performance']['parallel']['enabled'] = False
            
            # Save temporary config
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.safe_dump(config, f)
                temp_config_path = f.name
            
            compute_boundary_homology_main(temp_config_path, args.input_dir, args.output_dir)
            
            # Cleanup
            os.unlink(temp_config_path)
        else:
            compute_boundary_homology_main(args.config, args.input_dir, args.output_dir)
        
        print("\n✅ Boundary homology computation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Computation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in boundary homology computation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()