#!/bin/bash

# Set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create results directory if it doesn't exist
mkdir -p results/optimization

# Check if we're already in conda environment
if [[ "$CONDA_DEFAULT_ENV" == "myenv" ]]; then
    echo "Already in myenv environment"
else
    # Try to activate conda environment
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)" 2>/dev/null || true
        conda activate myenv 2>/dev/null || true
    fi
fi

# Run the interactive optimization tool
python -m src.utils.hyperparameter_optimization