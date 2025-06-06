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
    # Cleanup any hanging processes
    pkill -f pytest
    exit 1
}

# Function to cleanup on exit
cleanup() {
    print_message "Cleaning up..." "$YELLOW"
    pkill -f pytest
    conda deactivate
}

# Set up trap for cleanup
trap cleanup EXIT

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

# Set environment variables to help with test collection
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS="--tb=short"

# Run tests with timeout and specific options to prevent hanging
print_message "Running tests..." "$YELLOW"
pytest tests/ -v \
    --durations=10 \
    --maxfail=1 \
    --tb=short \
    --capture=no \
    --disable-pytest-warnings \
    --no-header \
    --no-summary \
    --showlocals

# Check if tests passed
if [ $? -eq 0 ]; then
    print_message "All tests passed successfully!" "$GREEN"
else
    print_message "Some tests failed!" "$RED"
    exit 1
fi

# Deactivate environment
conda deactivate 