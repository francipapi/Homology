# Decision Boundary Visualization Guide

This guide explains how to visualize decision boundaries computed by `decision_boundary_trainer.py`.

## Overview

The decision boundary visualization system provides comprehensive 3D visualizations of neural network decision boundaries extracted during training. It includes:

- **3D isosurface rendering** with interactive controls
- **Training evolution animations** showing how boundaries change
- **Statistical analysis** of boundary properties
- **Export capabilities** for further analysis

## Quick Start

### 1. Generate Decision Boundaries

First, train a model with boundary extraction enabled:

```bash
/opt/anaconda3/envs/myenv/bin/python src/models/decision_boundary_trainer.py
```

This creates boundary data in `results/decision_boundary_analysis/`.

### 2. Create Visualizations

Run the visualization script:

```bash
# Create all visualizations
/opt/anaconda3/envs/myenv/bin/python visualize_decision_boundaries.py --all

# Or specific visualizations
/opt/anaconda3/envs/myenv/bin/python visualize_decision_boundaries.py --animation --fps 5
```

### 3. View Results

Open the generated HTML files in your browser:
- `final_boundary.html` - Final decision boundary
- `boundary_evolution_animation.html` - Animated evolution
- `boundary_*_stage.html` - Early/middle/late stage snapshots

## Visualization Types

### Single Boundary Plot
Shows the decision boundary at a specific epoch with:
- 3D isosurface mesh (if available)
- Boundary point cloud colored by distance from origin
- **Training dataset overlay** showing both classes with distinct colors
- Interactive 3D controls (zoom, rotate, pan)

### Evolution Animation
Animated visualization showing:
- How the decision boundary evolves during training
- **Training dataset context** visible throughout animation
- Smooth transitions between epochs
- Playback controls and epoch slider

### Statistical Analysis
The script generates `boundary_evolution_stats.csv` with:
- Number of boundary points per epoch
- Mean/std distance from origin
- Mesh complexity metrics

## Configuration

Visualization settings are controlled by `configs/decision_boundary_config.yaml`:

```yaml
visualization:
  boundary_viz:
    show_mesh: true          # Show isosurface mesh
    show_points: true        # Show boundary points
    opacity: 0.7            # Mesh opacity
    color_scheme: 'viridis' # Color scheme
```

## Data Structure

The visualization expects:
- **Topology files**: `topology/topology_epoch_*.pt` containing boundary points
- **Mesh files**: `boundaries/boundary_epoch_*.ply` containing isosurface meshes

## Advanced Usage

### Custom Visualization

```python
from src.visualization.decision_boundary_viz import DecisionBoundaryVisualizer

# Load and visualize
visualizer = DecisionBoundaryVisualizer(config)
visualizer.load_boundary_data_from_directory('results/decision_boundary_analysis')

# Create custom plot
result = visualizer.boundary_data[-1]  # Final epoch
fig = visualizer.create_single_boundary_plot(result, title="Custom Title")
visualizer.save_plot(fig, "custom_boundary.html")
```

### Comparing Architectures

To compare decision boundaries from different architectures:

1. Train multiple models with different configs
2. Load results into separate visualizers
3. Use comparison methods (future feature)

## Troubleshooting

### No boundary data found
- Ensure training was run with `extraction_schedule.enabled: true`
- Check that files exist in `results/decision_boundary_analysis/`

### Large file sizes
- Animation files can be large (80+ MB)
- Reduce grid resolution or number of epochs to decrease size

### Missing mesh data
- Ensure `scikit-image` is installed for marching cubes
- Check that `isosurface.enabled: true` in config

## Dataset Overlay Feature

The visualization now includes **training dataset overlay** that shows:

### Color Coding:
- **Dodger Blue**: Class 0 points (inside torus) - highly visible with dark borders
- **Crimson**: Class 1 points (outside torus) - highly visible with dark borders
- **Viridis/Plasma**: Decision boundary points colored by distance (more subtle)

### Benefits:
- **Context Understanding**: See how the boundary relates to actual training data
- **Classification Accuracy**: Visually assess how well the boundary separates classes
- **Overfitting Detection**: Identify if boundary is too complex for the data distribution
- **Learning Progress**: Watch how boundary moves to better separate classes during training

### Visual Hierarchy:
The visualization prioritizes visual elements for better clarity:
1. **Training Dataset** (most prominent): Large, bright, opaque points with borders
2. **Decision Boundary Points** (secondary): Smaller, semi-transparent, color-coded by distance
3. **Boundary Surface** (background): Semi-transparent mesh providing context

### Dataset Loading:
The system automatically:
1. Tries to load dataset from saved training results
2. Falls back to generating dataset from training config
3. Handles both 2D and 3D coordinate data
4. Supports different torus configurations (solid, hollow, various radii)

## Key Insights from Visualizations

The visualizations with dataset overlay reveal:
1. **Boundary simplification**: Points decrease from 13,916 to 2,863 (79.4% reduction)
2. **Spatial concentration**: Mean distance decreases from 13.08 to 8.98
3. **Variance reduction**: Standard deviation decreases from 4.49 to 2.83
4. **Classification improvement**: Boundary becomes better aligned with data distribution

This indicates the network learns a more focused, consistent decision boundary that better separates the two classes as training progresses.