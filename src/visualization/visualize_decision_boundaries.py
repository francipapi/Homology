#!/usr/bin/env python3
"""
Main script for visualizing decision boundaries computed by decision_boundary_trainer.py

This script provides multiple visualization options:
1. Single boundary visualization at a specific epoch
2. Evolution animation showing boundary changes during training
3. Side-by-side comparison of different architectures
4. Statistical analysis of boundary properties

Usage:
    python visualize_decision_boundaries.py [options]
"""

import argparse
import sys
import os
sys.path.append('.')

from pathlib import Path
import numpy as np
import pandas as pd
from src.visualization.decision_boundary_viz import DecisionBoundaryVisualizer, load_boundary_config


def analyze_boundary_evolution(visualizer):
    """Analyze how decision boundary evolves during training."""
    print("\n=== Decision Boundary Evolution Analysis ===")
    
    results = []
    for result in visualizer.boundary_data:
        stats = {
            'epoch': result.epoch,
            'num_boundary_points': 0,
            'num_mesh_vertices': 0,
            'num_mesh_faces': 0,
            'mean_distance': 0,
            'std_distance': 0,
            'min_distance': 0,
            'max_distance': 0
        }
        
        if result.boundary_points is not None:
            distances = np.linalg.norm(result.boundary_points, axis=1)
            stats['num_boundary_points'] = len(result.boundary_points)
            stats['mean_distance'] = distances.mean()
            stats['std_distance'] = distances.std()
            stats['min_distance'] = distances.min()
            stats['max_distance'] = distances.max()
        
        if result.mesh_vertices is not None:
            stats['num_mesh_vertices'] = len(result.mesh_vertices)
            stats['num_mesh_faces'] = len(result.mesh_faces)
        
        results.append(stats)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results).sort_values('epoch')
    
    print("\nBoundary Point Statistics:")
    print(df[['epoch', 'num_boundary_points', 'mean_distance', 'std_distance']].to_string(index=False))
    
    print("\nMesh Complexity:")
    print(df[['epoch', 'num_mesh_vertices', 'num_mesh_faces']].to_string(index=False))
    
    # Analyze trends
    print("\nTrends:")
    print(f"- Boundary points: {df['num_boundary_points'].iloc[0]:,} → {df['num_boundary_points'].iloc[-1]:,} "
          f"({(df['num_boundary_points'].iloc[-1] / df['num_boundary_points'].iloc[0] - 1) * 100:.1f}% change)")
    print(f"- Mean distance: {df['mean_distance'].iloc[0]:.2f} → {df['mean_distance'].iloc[-1]:.2f}")
    print(f"- Distance spread (std): {df['std_distance'].iloc[0]:.2f} → {df['std_distance'].iloc[-1]:.2f}")
    
    return df


def create_all_visualizations(args):
    """Create all visualization types."""
    print(f"\n=== Loading Decision Boundary Data ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Config file: {args.config}")
    
    # Load configuration
    config = load_boundary_config(args.config)
    
    # Create visualizer
    visualizer = DecisionBoundaryVisualizer(config)
    
    # Load data
    if Path(args.data_dir).is_file():
        visualizer.load_boundary_results(args.data_dir)
    else:
        visualizer.load_boundary_data_from_directory(args.data_dir)
    
    # Load or generate dataset for overlay
    print("Loading training dataset...")
    visualizer.load_or_generate_dataset()
    
    print(f"\nLoaded {len(visualizer.boundary_data)} boundary snapshots")
    
    if not visualizer.boundary_data:
        print("ERROR: No boundary data found!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Analyze boundary evolution
    stats_df = analyze_boundary_evolution(visualizer)
    stats_df.to_csv(output_dir / "boundary_evolution_stats.csv", index=False)
    print(f"\nStatistics saved to: {output_dir / 'boundary_evolution_stats.csv'}")
    
    # 2. Create single boundary visualization (final epoch)
    if args.single or args.all:
        print("\n=== Creating Single Boundary Visualization ===")
        final_result = max(visualizer.boundary_data, key=lambda x: x.epoch)
        
        fig = visualizer.create_single_boundary_plot(
            final_result, 
            title=f"Decision Boundary at Epoch {final_result.epoch}"
        )
        
        output_path = output_dir / "final_boundary.html"
        visualizer.save_plot(fig, str(output_path))
        print(f"Saved: {output_path}")
        
        # Also create visualizations for early, middle, and late stages
        epochs = sorted([r.epoch for r in visualizer.boundary_data])
        stages = {
            'early': epochs[min(1, len(epochs)-1)],
            'middle': epochs[len(epochs)//2],
            'late': epochs[-2] if len(epochs) > 1 else epochs[-1]
        }
        
        for stage_name, epoch in stages.items():
            result = next(r for r in visualizer.boundary_data if r.epoch == epoch)
            fig = visualizer.create_single_boundary_plot(
                result,
                title=f"Decision Boundary - {stage_name.capitalize()} Stage (Epoch {epoch})"
            )
            output_path = output_dir / f"boundary_{stage_name}_stage.html"
            visualizer.save_plot(fig, str(output_path))
            print(f"Saved: {output_path}")
    
    # 3. Create evolution animation
    if args.animation or args.all:
        print("\n=== Creating Evolution Animation ===")
        
        # Adjust animation settings
        visualizer.viz_config.animation_fps = args.fps
        visualizer.viz_config.smooth_transitions = True
        
        fig = visualizer.create_evolution_animation()
        output_path = output_dir / "boundary_evolution_animation.html"
        visualizer.save_plot(fig, str(output_path))
        print(f"Saved: {output_path}")
    
    # 4. Create topology evolution plot (if betti numbers available)
    if args.topology or args.all:
        print("\n=== Creating Topology Evolution Plot ===")
        
        # Check if any results have betti numbers
        has_topology = any(hasattr(r, 'betti_numbers') and r.betti_numbers is not None 
                          for r in visualizer.boundary_data)
        
        if has_topology:
            fig = visualizer.create_topology_evolution_plot()
            output_path = output_dir / "topology_evolution.html"
            visualizer.save_plot(fig, str(output_path))
            print(f"Saved: {output_path}")
        else:
            print("No topology data (Betti numbers) found in results.")
            print("To compute topology, run: python src/topology/compute_boundary_homology.py")
    
    # 5. Export data for external analysis
    if args.export:
        print("\n=== Exporting Visualization Data ===")
        export_path = output_dir / "boundary_data_export.npz"
        visualizer.export_visualization_data(str(export_path))
        print(f"Exported data to: {export_path}")
    
    print("\n=== Visualization Complete ===")
    print(f"All outputs saved to: {output_dir}")
    
    # Print summary of what was created
    print("\nCreated files:")
    for file in sorted(output_dir.glob("*")):
        if file.is_file():
            size = file.stat().st_size / 1024 / 1024  # MB
            print(f"  - {file.name} ({size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize decision boundaries from neural network training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all visualizations
  python visualize_decision_boundaries.py --all
  
  # Create only animation with custom FPS
  python visualize_decision_boundaries.py --animation --fps 5
  
  # Create single boundary visualization
  python visualize_decision_boundaries.py --single
  
  # Use custom config and output directory
  python visualize_decision_boundaries.py --config my_config.yaml --output-dir my_results/
        """
    )
    
    # Input/output arguments
    parser.add_argument('--data-dir', type=str, 
                       default='results/decision_boundary_analysis',
                       help='Directory containing boundary data or path to results file')
    parser.add_argument('--config', type=str,
                       default='configs/decision_boundary_config.yaml',
                       help='Path to visualization config file')
    parser.add_argument('--output-dir', type=str,
                       default='results/decision_boundary_analysis/visualizations',
                       help='Output directory for visualizations')
    
    # Visualization options
    parser.add_argument('--all', action='store_true',
                       help='Create all visualization types')
    parser.add_argument('--single', action='store_true',
                       help='Create single boundary visualization')
    parser.add_argument('--animation', action='store_true',
                       help='Create evolution animation')
    parser.add_argument('--topology', action='store_true',
                       help='Create topology evolution plot')
    parser.add_argument('--export', action='store_true',
                       help='Export data for external analysis')
    
    # Animation settings
    parser.add_argument('--fps', type=int, default=3,
                       help='Frames per second for animation')
    
    args = parser.parse_args()
    
    # Default to all if no specific option selected
    if not any([args.all, args.single, args.animation, args.topology]):
        args.all = True
    
    # Run visualization
    create_all_visualizations(args)


if __name__ == "__main__":
    main()