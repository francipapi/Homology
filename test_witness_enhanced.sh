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

# Activate environment
print_message "Activating conda environment myenv..." "$YELLOW"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate myenv

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Test the enhanced witness complex script
print_message "Running enhanced witness complex script..." "$YELLOW"
python src/utils/compute_torus_homology_witness.py --no-save --no-parallel

# Deactivate environment
conda deactivate