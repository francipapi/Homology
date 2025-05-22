#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${2}${1}${NC}"
}

# Function to handle errors
handle_error() {
    print_message "Error: $1" "$RED"
    exit 1
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    handle_error "Conda is not installed"
fi

# Check if environment exists
if ! conda env list | grep -q "myenv"; then
    print_message "Creating conda environment..." "$YELLOW"
    conda create -y -n myenv python=3.9
fi

# Activate environment and install dependencies
print_message "Activating conda environment..." "$YELLOW"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate myenv

print_message "Installing dependencies..." "$YELLOW"
pip install -r requirements.txt

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check if config files exist
if [ ! -f "configs/training_config.yaml" ]; then
    handle_error "training_config.yaml not found in configs directory"
fi
if [ ! -f "configs/homology_config.yaml" ]; then
    handle_error "homology_config.yaml not found in configs directory"
fi
if [ ! -f "configs/visualization_config.yaml" ]; then
    handle_error "visualization_config.yaml not found in configs directory"
fi

# Create results directories if they don't exist
mkdir -p results/models results/plots results/homology

# Run main.py
print_message "Running main.py..." "$YELLOW"
python main.py

# Check if main.py executed successfully
if [ $? -eq 0 ]; then
    print_message "Pipeline completed successfully!" "$GREEN"
else
    print_message "Pipeline failed!" "$RED"
    exit 1
fi

# Deactivate environment
conda deactivate 