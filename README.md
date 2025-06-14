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

The project offers multiple approaches for training neural networks, primarily configured via `configs/training_config.yaml`. Key model definitions, such as the Multi-Layer Perceptron (MLP), are typically found in `src/models/torch_mlp.py`.

All training processes are designed to integrate with the broader pipeline of activation extraction and topological analysis. Device support (CPU, CUDA, Apple MPS) can be specified in the configuration file.

#### Standard PyTorch Training
    This is the primary approach for training individual PyTorch models.
    - **Script:** The core logic is typically handled by `src/models/trainer.py` (as utilized by `main.py` for the main pipeline).
        - *Note: The source code for `src/models/trainer.py` is not directly viewable by the assistant, so this description is based on its usage and project documentation.*
    - **Model:** It trains MLP architectures defined in `src/models/torch_mlp.py`.
    - **Configuration:** Highly configurable through `configs/training_config.yaml`, allowing adjustments to model architecture (layers, dimensions, activation functions), learning parameters (epochs, batch size, optimizer, learning rate), early stopping, and other training aspects.
    - **Features:** Supports functionalities like early stopping, model checkpointing (logic may reside within the trainer or be configured), and extraction of ReLU activations, which are crucial for the subsequent topological analysis.
    - **Device Support:** Can be run on CPU, CUDA, or Apple MPS, as specified in the `device` setting within the training configuration.

#### Parallel PyTorch Training (CPU)
    This approach allows for training multiple instances of the MLP model in parallel, primarily leveraging CPU cores.
    - **Script:** Implemented in `src/models/torch_parallel.py`, which contains the `ParallelTrainer` class.
    - **Mechanism:** Utilizes Python's `multiprocessing` library (specifically `ProcessPoolExecutor`) to distribute the training of individual network instances across different processes.
    - **Configuration:**
        - The number of networks to train in parallel is set by the `num_networks` parameter in `configs/training_config.yaml`.
        - The maximum number of parallel worker processes can be specified via `max_parallel_workers` in the same configuration file; if not set, it defaults to the number of CPU cores.
    - **Use Case:** Useful for experiments requiring multiple model runs with slight variations or for building ensembles, especially when GPU resources are limited or not the primary focus for this type of parallelization.
    - **Output:** Each network is trained independently, and results (like final accuracy and optionally saved models) are aggregated. Layer activation extraction for each network is also supported.

#### Vectorized PyTorch Training
    This method provides an efficient way to train an ensemble of neural networks simultaneously on a single device by leveraging PyTorch's vectorization capabilities.
    - **Script:** Implemented in `src/models/torch_vectorized.py`, featuring the `VectorizedTrainer` class.
    - **Mechanism:**
        - Converts the PyTorch MLP model (from `src/models/torch_mlp.py`) into a functional form using `make_functional_with_buffers` (from `torch.func` or `functorch`).
        - Uses `vmap` to apply operations (forward pass, loss calculation) across multiple model instances (an ensemble) in a vectorized manner. This means a single batch of data is processed by all models in the ensemble concurrently.
    - **Configuration:**
        - The number of models in the ensemble is controlled by `num_networks` in `configs/training_config.yaml`.
    - **Device Support:** Supports execution on CPU, CUDA, or Apple MPS. The vectorization occurs on the selected device, allowing for efficient parallel computation across the ensemble.
    - **Use Case:** Ideal for training multiple models with the same architecture but potentially different initializations, or for tasks where ensembling is beneficial, while efficiently utilizing hardware resources.
    - **Output:** Trains all models in the ensemble and can extract layer activations for each model.

#### Apple MLX Training
    The project includes support for Apple's MLX framework, designed for efficient machine learning on Apple Silicon (M1, M2, M3 series chips).
    - **Framework:** Leverages the MLX library to define and train models, optimizing performance on Apple hardware.
    - **Scripts:** Several MLX-based trainers are available, including (based on project structure and documentation):
        - `src/models/trainer_mlx.py` (or its updated versions like `trainer_mlx_v2.py`): For standard training of models using MLX.
        - `src/models/trainer_mlx_parallel.py`: Likely provides capabilities for parallel training of MLX models.
        - `src/models/vectorised_mlx.py`: May offer vectorized training similar to the PyTorch version but implemented in MLX.
        - *Note: The source code for these specific MLX trainer scripts was not directly viewable by the assistant. Their exact functionalities are inferred from naming conventions, project documentation (`CLAUDE.md`), and the presence of their compiled versions. For precise details, refer to the respective source files if available, or project maintainers.*
    - **Configuration:** Similar to PyTorch trainers, MLX training scripts are expected to be configurable via `configs/training_config.yaml`, allowing adjustments for model parameters, training hyperparameters, and data.
    - **Use Case:** Intended for users who wish to develop or train models specifically within the Apple MLX ecosystem, taking full advantage of Apple Silicon's unified memory architecture and performance characteristics.

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
2. Network Training: Train neural networks using the configured approach (see "Neural Network Training" section).
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
2. Configure your training session (model, hyperparameters, data, specific training script/method) in `configs/training_config.yaml` and then train the neural network using the chosen approach (e.g., standard PyTorch, parallel, vectorized, or MLX).
3. Extract activations and compute homology
4. Visualize results using plotting utilities

## Notes
- As detailed in the "Neural Network Training" section, several distinct training scripts and methodologies (Standard PyTorch, Parallel PyTorch, Vectorized PyTorch, Apple MLX) are available, each catering to different needs and hardware capabilities.
- All training approaches are primarily configured via `configs/training_config.yaml`. Refer to this file and the relevant training scripts for specific parameters.
- Comprehensive device support is available, including CPU, NVIDIA GPUs (CUDA), and Apple Silicon GPUs (MPS), configurable through the training configuration file.
