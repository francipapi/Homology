#!/usr/bin/env python3
"""
Script to analyze gradient-based similarity between neural networks.

This script demonstrates how to use the GradientSimilarityAnalyzer to compute
various similarity measures based on optimization dynamics, loss landscapes,
and gradient flow topology.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import yaml
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.gradient_similarity import GradientSimilarityAnalyzer
from src.models.torch_mlp import MLP, generate_torus_data
from src.models.torch_custom import CustomNet


def create_model(config: dict, model_type: str = 'mlp') -> nn.Module:
    """Create a model based on configuration."""
    if model_type == 'custom':
        return CustomNet(config['custom_architecture'])
    else:
        model_config = config['model']
        return MLP(
            input_dim=model_config['input_dim'],
            num_hidden_layers=model_config['num_hidden_layers'],
            hidden_dim=model_config['hidden_dim'],
            output_dim=model_config['output_dim'],
            activation_fn_name=model_config['activation_fn_name'],
            dropout_rate=model_config.get('dropout_rate', 0.0),
            use_batch_norm=model_config.get('use_batch_norm', False)
        )


def prepare_data(config: dict, device: torch.device) -> tuple:
    """Prepare data loaders based on configuration."""
    data_config = config['data']
    
    # Generate synthetic data
    X, y = generate_torus_data(
        n=data_config.get('generation', {}).get('n', 1000),
        big_radius=data_config.get('generation', {}).get('big_radius', 3),
        small_radius=data_config.get('generation', {}).get('small_radius', 1),
        solid=data_config.get('generation', {}).get('solid', False),
        interior_noise=data_config.get('generation', {}).get('interior_noise', 0.1)
    )
    
    # Move to device
    X = X.to(device)
    y = y.to(device).float()
    
    # Split data
    split_ratio = data_config.get('split_ratio', 0.8)
    train_size = int(split_ratio * len(X))
    
    # Shuffle
    perm = torch.randperm(len(X))
    X = X[perm]
    y = y[perm]
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def analyze_single_model(model: nn.Module, train_loader: DataLoader, 
                        analyzer: GradientSimilarityAnalyzer,
                        loss_fn: nn.Module, args: argparse.Namespace) -> dict:
    """Analyze a single model's gradient properties."""
    results = {}
    
    print("\n" + "="*50)
    print("Analyzing Model Gradient Properties")
    print("="*50)
    
    # 1. Track gradient flow
    if args.track_gradient_flow:
        print("\n1. Tracking Gradient Flow...")
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        flow = analyzer.track_gradient_flow(
            model, loss_fn, train_loader, optimizer, num_steps=args.num_flow_steps
        )
        results['gradient_flow'] = flow
        print(f"   Tracked {len(flow)} gradient flow snapshots")
        
        # Visualize if requested
        if args.visualize:
            save_path = Path(args.output_dir) / 'gradient_flow.png'
            analyzer.visualize_gradient_flow(flow, save_path=str(save_path))
            print(f"   Saved visualization to {save_path}")
    
    # 2. Analyze loss landscape
    if args.analyze_landscape:
        print("\n2. Analyzing Loss Landscape...")
        landscape = analyzer.analyze_loss_landscape(
            model, loss_fn, train_loader, 
            resolution=args.landscape_resolution,
            epsilon=args.landscape_epsilon
        )
        results['loss_landscape'] = landscape
        
        print(f"   Landscape roughness: {landscape['roughness']:.4f}")
        print(f"   Landscape convexity: {landscape['convexity']:.4f}")
        print(f"   Basin volume: {landscape['basin_volume']:.4f}")
        print(f"   Mean barrier height: {np.mean(landscape['barrier_heights']):.4f}")
        
        # Visualize if requested
        if args.visualize:
            save_path = Path(args.output_dir) / 'loss_landscape.png'
            analyzer.visualize_loss_landscape(landscape, save_path=str(save_path))
            print(f"   Saved visualization to {save_path}")
    
    # 3. Analyze curvature
    if args.analyze_curvature:
        print("\n3. Analyzing Curvature Properties...")
        curvature = analyzer.analyze_curvature(
            model, loss_fn, train_loader,
            num_directions=args.num_curvature_directions
        )
        results['curvature'] = curvature
        
        print(f"   Mean curvature: {curvature['mean_curvature']:.4f}")
        print(f"   Max curvature: {curvature['max_curvature']:.4f}")
        print(f"   Min curvature: {curvature['min_curvature']:.4f}")
        print(f"   Negative curvature ratio: {curvature['negative_curvature_ratio']:.4f}")
    
    # 4. Analyze critical points
    if args.analyze_critical_points:
        print("\n4. Analyzing Critical Points...")
        critical = analyzer.analyze_critical_points(
            model, loss_fn, train_loader,
            num_perturbations=args.num_perturbations
        )
        results['critical_points'] = critical
        
        print(f"   Number of minima found: {critical['num_minima']}")
        print(f"   Number of saddles found: {critical['num_saddles']}")
        print(f"   Average Morse index: {critical['average_morse_index']:.2f}")
    
    # 5. Track optimization dynamics
    if args.track_dynamics:
        print("\n5. Tracking Optimization Dynamics...")
        dynamics = analyzer.track_optimization_dynamics(
            model, loss_fn, train_loader,
            optimizer_class=optim.Adam,
            lr=args.learning_rate,
            num_epochs=args.num_dynamics_epochs
        )
        results['optimization_dynamics'] = dynamics
        
        print(f"   Final loss: {dynamics['loss'][-1]:.4f}")
        print(f"   Final gradient norm: {dynamics['gradient_norm'][-1]:.4f}")
        print(f"   Average effective LR: {np.mean(dynamics['effective_lr']):.4f}")
    
    return results


def compare_models(model1: nn.Module, model2: nn.Module,
                  train_loader: DataLoader, analyzer: GradientSimilarityAnalyzer,
                  loss_fn: nn.Module, args: argparse.Namespace) -> dict:
    """Compare two models using gradient-based similarity measures."""
    print("\n" + "="*50)
    print("Comparing Models")
    print("="*50)
    
    similarities = {}
    
    # 1. Compare gradient flows
    if args.compare_gradient_flows:
        print("\n1. Comparing Gradient Flows...")
        
        # Track flows for both models
        optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate)
        optimizer2 = optim.Adam(model2.parameters(), lr=args.learning_rate)
        
        flow1 = analyzer.track_gradient_flow(
            model1, loss_fn, train_loader, optimizer1, num_steps=args.num_flow_steps
        )
        flow2 = analyzer.track_gradient_flow(
            model2, loss_fn, train_loader, optimizer2, num_steps=args.num_flow_steps
        )
        
        # Compare using different methods
        for method in ['trajectory', 'velocity', 'curvature']:
            similarity = analyzer.compute_gradient_flow_similarity(flow1, flow2, method=method)
            similarities[f'gradient_flow_{method}'] = similarity
            print(f"   {method.capitalize()} similarity: {similarity:.4f}")
    
    # 2. Compare loss landscapes
    if args.compare_landscapes:
        print("\n2. Comparing Loss Landscapes...")
        
        landscape1 = analyzer.analyze_loss_landscape(
            model1, loss_fn, train_loader,
            resolution=args.landscape_resolution,
            epsilon=args.landscape_epsilon
        )
        landscape2 = analyzer.analyze_loss_landscape(
            model2, loss_fn, train_loader,
            resolution=args.landscape_resolution,
            epsilon=args.landscape_epsilon
        )
        
        landscape_similarities = analyzer.compare_loss_landscapes(landscape1, landscape2)
        similarities.update(landscape_similarities)
        
        for key, value in landscape_similarities.items():
            print(f"   {key}: {value:.4f}")
    
    # 3. Compare Hessian properties
    if args.compare_hessian:
        print("\n3. Comparing Hessian Properties...")
        
        for method in ['eigenvalue', 'trace', 'determinant', 'condition']:
            similarity = analyzer.compute_hessian_similarity(
                model1, model2, loss_fn, train_loader,
                method=method, top_k=args.hessian_top_k
            )
            similarities[f'hessian_{method}'] = similarity
            print(f"   {method.capitalize()} similarity: {similarity:.4f}")
    
    # 4. Compare NTK
    if args.compare_ntk:
        print("\n4. Comparing Neural Tangent Kernels...")
        
        ntk_similarities = analyzer.compare_ntk_similarity(
            model1, model2, train_loader,
            num_samples=args.ntk_samples
        )
        similarities.update(ntk_similarities)
        
        for key, value in ntk_similarities.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
    
    # 5. Compare optimization dynamics
    if args.compare_dynamics:
        print("\n5. Comparing Optimization Dynamics...")
        
        dynamics1 = analyzer.track_optimization_dynamics(
            model1, loss_fn, train_loader,
            optimizer_class=optim.Adam,
            lr=args.learning_rate,
            num_epochs=args.num_dynamics_epochs
        )
        dynamics2 = analyzer.track_optimization_dynamics(
            model2, loss_fn, train_loader,
            optimizer_class=optim.Adam,
            lr=args.learning_rate,
            num_epochs=args.num_dynamics_epochs
        )
        
        dynamics_similarities = analyzer.compare_optimization_dynamics(dynamics1, dynamics2)
        similarities.update(dynamics_similarities)
        
        print(f"   Overall dynamics similarity: {dynamics_similarities['overall_dynamics_similarity']:.4f}")
    
    return similarities


def visualize_similarity_matrix(similarities_matrix: np.ndarray, 
                               save_path: Path, model_names: list = None):
    """Visualize similarity matrix as heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Create labels
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(similarities_matrix))]
    
    # Create heatmap
    sns.heatmap(similarities_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                xticklabels=model_names,
                yticklabels=model_names,
                square=True,
                cbar_kws={'label': 'Similarity'})
    
    plt.title('Gradient-Based Model Similarity Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze gradient-based similarity between neural networks')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--output-dir', type=str, default='results/gradient_analysis',
                       help='Output directory for results')
    
    # Model options
    parser.add_argument('--model-type', type=str, default='mlp',
                       choices=['mlp', 'custom'], help='Type of model to use')
    parser.add_argument('--num-models', type=int, default=2,
                       help='Number of models to compare')
    
    # Analysis options
    parser.add_argument('--track-gradient-flow', action='store_true',
                       help='Track gradient flow trajectory')
    parser.add_argument('--analyze-landscape', action='store_true',
                       help='Analyze loss landscape')
    parser.add_argument('--analyze-curvature', action='store_true',
                       help='Analyze curvature properties')
    parser.add_argument('--analyze-critical-points', action='store_true',
                       help='Analyze critical points')
    parser.add_argument('--track-dynamics', action='store_true',
                       help='Track optimization dynamics')
    
    # Comparison options
    parser.add_argument('--compare-gradient-flows', action='store_true',
                       help='Compare gradient flows between models')
    parser.add_argument('--compare-landscapes', action='store_true',
                       help='Compare loss landscapes')
    parser.add_argument('--compare-hessian', action='store_true',
                       help='Compare Hessian properties')
    parser.add_argument('--compare-ntk', action='store_true',
                       help='Compare Neural Tangent Kernels')
    parser.add_argument('--compare-dynamics', action='store_true',
                       help='Compare optimization dynamics')
    
    # Analysis parameters
    parser.add_argument('--num-flow-steps', type=int, default=50,
                       help='Number of gradient flow steps to track')
    parser.add_argument('--landscape-resolution', type=int, default=30,
                       help='Resolution for loss landscape analysis')
    parser.add_argument('--landscape-epsilon', type=float, default=0.1,
                       help='Perturbation range for landscape analysis')
    parser.add_argument('--num-curvature-directions', type=int, default=20,
                       help='Number of directions for curvature analysis')
    parser.add_argument('--num-perturbations', type=int, default=30,
                       help='Number of perturbations for critical point analysis')
    parser.add_argument('--hessian-top-k', type=int, default=50,
                       help='Number of top eigenvalues to compute')
    parser.add_argument('--ntk-samples', type=int, default=100,
                       help='Number of samples for NTK computation')
    parser.add_argument('--num-dynamics-epochs', type=int, default=5,
                       help='Number of epochs for dynamics tracking')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for optimization')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    # Quick mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced parameters')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick mode
    if args.quick:
        args.num_flow_steps = 10
        args.landscape_resolution = 20
        args.num_curvature_directions = 10
        args.num_perturbations = 10
        args.hessian_top_k = 20
        args.ntk_samples = 50
        args.num_dynamics_epochs = 2
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = GradientSimilarityAnalyzer(device=device)
    
    # Prepare data
    train_loader, test_loader = prepare_data(config, device)
    print(f"Data prepared: {len(train_loader)} training batches")
    
    # Setup loss function
    loss_fn = nn.BCELoss()
    
    # Create models
    models = []
    for i in range(args.num_models):
        model = create_model(config, args.model_type)
        model.to(device)
        models.append(model)
        print(f"Created model {i+1}")
    
    # Analyze individual models
    if any([args.track_gradient_flow, args.analyze_landscape, 
            args.analyze_curvature, args.analyze_critical_points,
            args.track_dynamics]):
        
        for i, model in enumerate(models):
            print(f"\n{'='*50}")
            print(f"Analyzing Model {i+1}")
            print(f"{'='*50}")
            
            results = analyze_single_model(
                model, train_loader, analyzer, loss_fn, args
            )
            
            # Save results
            results_path = output_dir / f'model_{i+1}_analysis.npz'
            np.savez(results_path, **{k: v for k, v in results.items() 
                                     if isinstance(v, (np.ndarray, list, dict))})
            print(f"\nSaved analysis results to {results_path}")
    
    # Compare models
    if args.num_models > 1 and any([args.compare_gradient_flows, args.compare_landscapes,
                                    args.compare_hessian, args.compare_ntk,
                                    args.compare_dynamics]):
        
        # Create similarity matrix
        n = len(models)
        similarity_matrix = np.eye(n)
        all_similarities = {}
        
        for i in range(n):
            for j in range(i+1, n):
                print(f"\n{'='*50}")
                print(f"Comparing Model {i+1} and Model {j+1}")
                print(f"{'='*50}")
                
                similarities = compare_models(
                    models[i], models[j], train_loader, analyzer, loss_fn, args
                )
                
                # Store detailed similarities
                all_similarities[f'{i}_{j}'] = similarities
                
                # Compute overall similarity (average of all metrics)
                numeric_similarities = [v for v in similarities.values() 
                                      if isinstance(v, (int, float))]
                overall_similarity = np.mean(numeric_similarities) if numeric_similarities else 0
                
                similarity_matrix[i, j] = overall_similarity
                similarity_matrix[j, i] = overall_similarity
        
        # Save similarity matrix
        matrix_path = output_dir / 'similarity_matrix.npy'
        np.save(matrix_path, similarity_matrix)
        print(f"\nSaved similarity matrix to {matrix_path}")
        
        # Save detailed similarities
        details_path = output_dir / 'detailed_similarities.npz'
        np.savez(details_path, **all_similarities)
        print(f"Saved detailed similarities to {details_path}")
        
        # Visualize similarity matrix
        if args.visualize:
            viz_path = output_dir / 'similarity_matrix.png'
            visualize_similarity_matrix(similarity_matrix, viz_path)
            print(f"Saved similarity matrix visualization to {viz_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("SIMILARITY MATRIX SUMMARY")
        print("="*50)
        print(similarity_matrix)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()