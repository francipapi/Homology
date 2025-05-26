# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project analyzes the topological properties of neural networks during training by computing persistent homology on layer activations. The pipeline consists of dataset generation, neural network training, activation extraction, and topological analysis using persistent homology.

## Key Commands

### Running the Main Pipeline
```bash
# Full pipeline execution (creates conda env, installs deps, runs main.py)
./run_main.sh

# Direct execution (requires manual environment setup)
python main.py
```

### Testing
```bash
# Run all tests with proper environment setup
./run_tests.sh

# Run specific test
pytest tests/test_pipeline.py -v
```

### Environment Setup
```bash
# Create conda environment with dependencies
conda create -y -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Architecture Overview

### Core Pipeline Flow
1. **Data Generation** (`src/data/`): Creates synthetic torus datasets
2. **Neural Network Training** (`src/models/`): Trains networks and extracts ReLU activations
3. **Topological Analysis** (`src/topology/`): Computes persistent homology on activations
4. **Visualization** (`src/visualization/`): Plots Betti curves and results

### Multiple Training Implementations
The project supports several training frameworks:
- **PyTorch**: `src/models/trainer.py`, `torch_mlp.py`, `torch_vectorized.py` 
- **Apple MLX**: `trainer_mlx.py`, `trainer_mlx_parallel.py` for Apple Silicon optimization
- **Parallel Training**: `torch_parallel.py` for training multiple networks simultaneously

### Configuration System
All parameters are controlled via YAML configs in `configs/`:
- `training_config.yaml`: Model architecture, training hyperparameters, data generation
- `homology_config.yaml`: Persistent homology computation parameters
- `visualization_config.yaml`: Plotting and visualization settings

### Key Functions to Know
- `src/data/dataset.generate()`: Creates torus pair datasets
- `src/models/trainer.train()`: Main training function that returns layer activations
- `src/topology/homology.compute_persistent_homology()`: Core homology computation
- `src/utils/graph.distance()`: Distance matrix computation for Rips complex construction

### Device Support
The codebase supports:
- CPU training
- CUDA (NVIDIA GPUs) 
- MPS (Apple Silicon GPUs via Metal Performance Shaders)
- MLX (Apple's ML framework for Apple Silicon)

Device selection is controlled via the `device` parameter in training configs ('auto', 'cpu', 'cuda', 'mps').

### Output Structure
Results are organized in `results/`:
- `models/`: Trained model checkpoints
- `plots/`: Dataset visualizations and Betti curves
- `homology/`: Persistence diagrams, barcodes, and Betti numbers per layer