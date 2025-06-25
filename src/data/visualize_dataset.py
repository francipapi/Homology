#!/usr/bin/env python3
"""
Simple script to visualize dataset with FPS applied using training_config.yaml parameters
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from visualize_dataset_3d import TorusDataset3DVisualizer

def main():
    # Use training config path
    config_path = 'configs/training_config.yaml'
    
    # Create visualizer - it will automatically load training config params
    visualizer = TorusDataset3DVisualizer()
    
    # Enable FPS visualization
    visualizer.config['fps']['enabled'] = True
    
    # Run visualization
    print("🚀 Starting dataset visualization with FPS...")
    visualizer.run_visualization(show_interactive=True)

if __name__ == "__main__":
    main()