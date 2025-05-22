# Neural Network Topological Analysis

This project analyzes the topological properties of neural networks during training by computing persistent homology on the activations of different layers. The pipeline consists of several key components:

## Project Structure

### Data Generation and Processing
- `data.py`: Contains functions for generating synthetic datasets (torus pairs) and data processing utilities
  - `generate()`: Creates complex torus pair datasets
  - `gen_easy()`: Creates simple torus pair datasets
  - `farthest_point_sampling()`: Implements FPS for point cloud sampling
  - `plot_torus_points()`: Visualizes 3D point clouds

### Neural Network Training
- `train.py`: Main training script for PyTorch models
  - Implements a configurable dense neural network
  - Supports early stopping and model checkpointing
  - Extracts ReLU activations for topological analysis
- `train2.py`: Alternative training implementation with different configurations
- `train_mlx.py` & `train2_mlx.py`: Apple MLX framework implementations

### Topological Analysis
- `homology.py`: Core homology computation module
  - `compute_persistent_homology()`: Computes persistent homology using Gudhi
  - `multi_hom()`: Parallel implementation for multiple computations
  - `multi_pers()`: Computes both persistence diagrams and Betti numbers
  - `wes_dist()`: Computes Wasserstein distance between persistence diagrams

### Visualization
- `plot_curves.py`: Visualization tools for topological analysis results
  - `plot_curves_from_h5()`: Plots Betti number curves with statistics
- `umaplot.py`: Additional visualization utilities

### Supporting Files
- `distance.py`: Distance computation utilities
- `mps.py`: Apple MPS (Metal Performance Shaders) specific implementations
- `multi_trackH.py`: Tracking homology changes during training
- `multi_hom.py`: Parallel homology computation utilities

## Pipeline Flow
1. Data Generation: Create synthetic datasets using `data.py`
2. Network Training: Train neural networks using `train.py` or variants
3. Activation Extraction: Extract layer activations during training
4. Topological Analysis: Compute persistent homology using `homology.py`
5. Visualization: Plot and analyze results using `plot_curves.py`

## Dependencies
- PyTorch
- NumPy
- Gudhi
- Matplotlib
- Plotly
- scikit-learn
- h5py

## Usage
1. Generate or load your dataset
2. Train the neural network using appropriate training script
3. Extract activations and compute homology
4. Visualize results using plotting utilities

## Notes
- Multiple implementations exist for some components (e.g., training scripts) with subtle differences in configuration and optimization
- The project supports both CPU and GPU (including Apple M1/M2) training
- Parallel processing is implemented for computationally intensive tasks 