#!/usr/bin/env python3
"""
Random Network Generator

This script generates a random neural network using the architecture specified in a YAML 
configuration file and saves it to the models directory. The network is initialized with 
random weights but not trained, making it useful for:

1. Creating baseline models for topological comparison
2. Studying the effect of random initialization on network topology
3. Generating control models for analysis

Usage:
    python src/models/generate_random_network.py configs/training_config.yaml
    python src/models/generate_random_network.py configs/ring_training_config.yaml --num-networks 5
"""

import torch
import torch.nn as nn
import yaml
import argparse
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.torch_mlp import MLP
from src.models.torch_custom import CustomNet


class RandomNetworkGenerator:
    """
    Generates random neural networks based on configuration specifications.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the generator with a configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        
        # Determine output directory
        self.output_dir = Path('results/models/random_networks')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Random Network Generator initialized")
        print(f"Configuration loaded from: {config_path}")
        print(f"Output directory: {self.output_dir}")
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducible network generation."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def create_random_mlp(self, network_id: Optional[int] = None) -> MLP:
        """
        Create a random MLP model using the standard architecture.
        
        Args:
            network_id: Optional identifier for the network
            
        Returns:
            Initialized MLP model with random weights
        """
        model = MLP(
            input_dim=self.model_config['input_dim'],
            num_hidden_layers=self.model_config['num_hidden_layers'],
            hidden_dim=self.model_config['hidden_dim'],
            output_dim=self.model_config['output_dim'],
            activation_fn_name=self.model_config.get('activation_fn_name', 'relu'),
            output_activation_fn_name=self.model_config.get('output_activation_fn_name', 'sigmoid'),
            dropout_rate=self.model_config.get('dropout_rate', 0.0),
            use_batch_norm=self.model_config.get('use_batch_norm', False)
        )
        
        # Apply custom weight initialization if specified
        self._apply_custom_initialization(model, network_id)
        
        return model
    
    def create_random_custom_net(self, network_id: Optional[int] = None) -> CustomNet:
        """
        Create a random custom architecture model.
        
        Args:
            network_id: Optional identifier for the network
            
        Returns:
            Initialized CustomNet model with random weights
        """
        custom_config = self.config.get('custom_architecture', {})
        if not custom_config.get('enabled', False):
            raise ValueError("Custom architecture is not enabled in the configuration")
        
        model = CustomNet(custom_config)
        
        # Apply custom weight initialization if specified
        self._apply_custom_initialization(model, network_id)
        
        return model
    
    def _apply_custom_initialization(self, model: nn.Module, network_id: Optional[int] = None):
        """
        Apply custom weight initialization schemes to the model.
        
        Args:
            model: The neural network model
            network_id: Optional network identifier for varied initialization
        """
        init_config = self.config.get('initialization', {})
        init_method = init_config.get('method', 'default')
        
        if init_method == 'default':
            # Use the model's built-in initialization (already applied)
            pass
        elif init_method == 'xavier_uniform':
            self._apply_xavier_uniform(model)
        elif init_method == 'xavier_normal':
            self._apply_xavier_normal(model)
        elif init_method == 'kaiming_uniform':
            self._apply_kaiming_uniform(model)
        elif init_method == 'kaiming_normal':
            self._apply_kaiming_normal(model)
        elif init_method == 'orthogonal':
            self._apply_orthogonal(model)
        elif init_method == 'random_scaled':
            scale = init_config.get('scale', 1.0)
            if network_id is not None:
                # Vary scale slightly for each network
                scale *= (1.0 + 0.1 * np.sin(network_id))
            self._apply_random_scaled(model, scale)
        else:
            print(f"Warning: Unknown initialization method '{init_method}', using default")
    
    def _apply_xavier_uniform(self, model: nn.Module):
        """Apply Xavier uniform initialization."""
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def _apply_xavier_normal(self, model: nn.Module):
        """Apply Xavier normal initialization."""
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def _apply_kaiming_uniform(self, model: nn.Module):
        """Apply Kaiming uniform initialization."""
        for layer in model.modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def _apply_kaiming_normal(self, model: nn.Module):
        """Apply Kaiming normal initialization."""
        for layer in model.modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def _apply_orthogonal(self, model: nn.Module):
        """Apply orthogonal initialization."""
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def _apply_random_scaled(self, model: nn.Module, scale: float):
        """Apply scaled random initialization."""
        for layer in model.modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.uniform_(layer.weight, -scale, scale)
                if layer.bias is not None:
                    nn.init.uniform_(layer.bias, -scale/10, scale/10)
    
    def generate_single_network(self, network_id: int = 0, seed: Optional[int] = None, 
                               force_architecture: Optional[str] = None) -> str:
        """
        Generate a single random network and save it.
        
        Args:
            network_id: Identifier for the network
            seed: Random seed for reproducible generation
            force_architecture: Force architecture type ('mlp' or 'custom'), overrides config
            
        Returns:
            Path to the saved network file
        """
        if seed is not None:
            self.set_random_seed(seed)
        
        # Determine architecture type
        custom_config = self.config.get('custom_architecture', {})
        
        if force_architecture is not None:
            if force_architecture.lower() == 'custom':
                use_custom = True
                arch_type = 'custom'
            elif force_architecture.lower() == 'mlp':
                use_custom = False
                arch_type = 'mlp'
            else:
                raise ValueError(f"Invalid architecture type: {force_architecture}. Must be 'mlp' or 'custom'")
        else:
            # Use config to determine architecture
            use_custom = custom_config.get('enabled', False)
            arch_type = 'custom' if use_custom else 'mlp'
        
        if use_custom:
            model = self.create_random_custom_net(network_id)
        else:
            model = self.create_random_mlp(network_id)
        
        # Generate filename
        init_method = self.config.get('initialization', {}).get('method', 'default')
        filename = f"random_{arch_type}_net_{network_id:03d}_{init_method}"
        if seed is not None:
            filename += f"_seed_{seed}"
        filename += ".pth"
        
        filepath = self.output_dir / filename
        
        # Save the model
        save_data = {
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'network_id': network_id,
            'architecture_type': arch_type,
            'initialization_method': init_method,
            'random_seed': seed,
            'model_info': {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        }
        
        # Add architecture-specific information
        if use_custom:
            save_data['custom_config'] = custom_config
        else:
            save_data['model_config'] = self.model_config
        
        torch.save(save_data, filepath)
        
        print(f"Generated random network {network_id}: {filepath}")
        print(f"  Architecture: {arch_type}")
        print(f"  Initialization: {init_method}")
        print(f"  Parameters: {save_data['model_info']['total_parameters']:,}")
        
        return str(filepath)
    
    def generate_multiple_networks(self, num_networks: int, base_seed: int = 42, 
                                  force_architecture: Optional[str] = None) -> list:
        """
        Generate multiple random networks with different seeds.
        
        Args:
            num_networks: Number of networks to generate
            base_seed: Base seed for generation (each network gets base_seed + network_id)
            force_architecture: Force architecture type ('mlp' or 'custom'), overrides config
            
        Returns:
            List of paths to saved network files
        """
        arch_info = f" ({force_architecture})" if force_architecture else ""
        print(f"\nGenerating {num_networks} random networks{arch_info}...")
        print("=" * 50)
        
        filepaths = []
        
        for i in range(num_networks):
            seed = base_seed + i
            filepath = self.generate_single_network(
                network_id=i, 
                seed=seed, 
                force_architecture=force_architecture
            )
            filepaths.append(filepath)
        
        print("=" * 50)
        print(f"Generated {len(filepaths)} random networks in: {self.output_dir}")
        
        # Generate summary
        self._generate_summary(filepaths)
        
        return filepaths
    
    def _generate_summary(self, filepaths: list):
        """Generate a summary file with information about generated networks."""
        summary_file = self.output_dir / "generation_summary.yaml"
        
        summary_data = {
            'generation_info': {
                'total_networks': len(filepaths),
                'generation_timestamp': str(torch.utils.data.get_worker_info()),
                'config_used': self.config,
                'output_directory': str(self.output_dir)
            },
            'networks': []
        }
        
        for filepath in filepaths:
            # Load network info
            network_data = torch.load(filepath, map_location='cpu')
            summary_data['networks'].append({
                'filename': Path(filepath).name,
                'network_id': network_data['network_id'],
                'architecture_type': network_data['architecture_type'],
                'initialization_method': network_data['initialization_method'],
                'random_seed': network_data['random_seed'],
                'total_parameters': network_data['model_info']['total_parameters']
            })
        
        with open(summary_file, 'w') as f:
            yaml.dump(summary_data, f, default_flow_style=False, indent=2)
        
        print(f"Generation summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate random neural networks from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate a single random MLP network (overrides config)
    python src/models/generate_random_network.py configs/training_config.yaml --architecture mlp
    
    # Generate 10 random custom architecture networks
    python src/models/generate_random_network.py configs/ring_training_config.yaml --num-networks 10 --architecture custom
    
    # Generate with custom seed and initialization
    python src/models/generate_random_network.py configs/training_config.yaml --seed 12345 --init-method xavier_uniform --architecture mlp
    
    # Use config file architecture setting (default behavior)
    python src/models/generate_random_network.py configs/training_config.yaml --num-networks 5
        """
    )
    
    parser.add_argument(
        'config_path', 
        type=str, 
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '--num-networks', 
        type=int, 
        default=1, 
        help='Number of random networks to generate (default: 1)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Base random seed for generation (default: 42)'
    )
    parser.add_argument(
        '--init-method', 
        type=str, 
        choices=['default', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 
                'kaiming_normal', 'orthogonal', 'random_scaled'],
        help='Override initialization method from config'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        help='Override output directory (default: results/models/)'
    )
    parser.add_argument(
        '--architecture', 
        type=str, 
        choices=['mlp', 'custom'],
        help='Force architecture type (overrides config file setting)'
    )
    
    args = parser.parse_args()
    
    # Verify config file exists
    if not Path(args.config_path).exists():
        print(f"Error: Configuration file not found: {args.config_path}")
        return 1
    
    try:
        # Create generator
        generator = RandomNetworkGenerator(args.config_path)
        
        # Override output directory if specified
        if args.output_dir:
            generator.output_dir = Path(args.output_dir)
            generator.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Override initialization method if specified
        if args.init_method:
            if 'initialization' not in generator.config:
                generator.config['initialization'] = {}
            generator.config['initialization']['method'] = args.init_method
        
        # Generate networks
        if args.num_networks == 1:
            filepath = generator.generate_single_network(
                seed=args.seed,
                force_architecture=args.architecture
            )
            print(f"\nRandom network generated successfully: {filepath}")
        else:
            filepaths = generator.generate_multiple_networks(
                num_networks=args.num_networks,
                base_seed=args.seed,
                force_architecture=args.architecture
            )
            print(f"\n{len(filepaths)} random networks generated successfully!")
        
        return 0
        
    except Exception as e:
        print(f"Error generating random networks: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())