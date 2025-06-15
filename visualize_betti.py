#!/usr/bin/env python3
"""
Betti Curves Visualization Runner

Simple script to create Betti curves visualizations from computed homology results.
This script serves as a convenient entry point for generating plots.

Usage:
    python visualize_betti.py                          # Use defaults
    python visualize_betti.py --format pdf --dpi 600   # High-res PDF output
    python visualize_betti.py --help                   # Show options

Author: Generated for Homology project
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).resolve().parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

try:
    from src.visualization.betti_curves import BettiCurvesVisualizer, main
    
    if __name__ == "__main__":
        print("BETTI CURVES VISUALIZATION")
        print("=" * 40)
        print("Generating plots showing how Betti numbers change across neural network layers...")
        print()
        
        # Run the main visualization function
        main()
        
except ImportError as e:
    print(f"ERROR: Could not import visualization module: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Visualization failed: {e}")
    sys.exit(1)