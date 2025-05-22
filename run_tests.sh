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
#hello

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

# Run tests with timeout if available
if command -v gtimeout &> /dev/null; then
    print_message "Running tests with timeout..." "$YELLOW"
    gtimeout 300 pytest tests/ -v
elif command -v timeout &> /dev/null; then
    print_message "Running tests with timeout..." "$YELLOW"
    timeout 300 pytest tests/ -v
else
    print_message "Running tests without timeout..." "$YELLOW"
    pytest tests/ -v
fi

# Check if tests passed
if [ $? -eq 0 ]; then
    print_message "All tests passed successfully!" "$GREEN"
else
    print_message "Some tests failed!" "$RED"
    exit 1
fi

# Deactivate environment
conda deactivate 