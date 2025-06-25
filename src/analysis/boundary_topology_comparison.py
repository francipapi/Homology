"""
Boundary Topology Comparison and Analysis

This module provides comprehensive analysis tools for comparing decision boundary 
topologies across different neural network architectures. Features include:

- Statistical analysis of topology evolution
- Architecture comparison metrics
- Convergence and stability analysis
- Export capabilities for research and presentation

Author: Claude Code
Date: 2025
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import pairwise_distances
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import yaml
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import boundary extraction results
from src.topology.decision_boundary_homology import BoundaryExtractionResult, load_boundary_config
from src.visualization.decision_boundary_viz import DecisionBoundaryVisualizer


@dataclass
class TopologyMetrics:
    """Container for topology analysis metrics."""
    betti_stability: float
    convergence_epoch: Optional[int]
    final_topology: List[int]
    topology_complexity: float
    shape_variance: float
    architecture_label: str


@dataclass
class ComparisonResult:
    """Result of architecture comparison analysis."""
    architecture_labels: List[str]
    topology_metrics: List[TopologyMetrics]
    statistical_tests: Dict[str, Any]
    similarity_matrix: np.ndarray
    convergence_comparison: Dict[str, Any]
    stability_comparison: Dict[str, Any]


class BoundaryTopologyAnalyzer:
    """
    Comprehensive analysis system for decision boundary topology.
    
    This class provides methods for:
    1. Computing topology stability metrics
    2. Analyzing convergence properties
    3. Comparing architectures statistically
    4. Generating comprehensive reports
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the analyzer.
        
        Parameters:
        - config: Analysis configuration dictionary
        """
        self.config = config or {}
        self.analysis_config = self.config.get('analysis', {})
        
        # Storage for analysis data
        self.architecture_data = {}  # Dict mapping architecture labels to boundary data
        self.comparison_results = None
        
        # Configure plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def add_architecture_data(self, label: str, boundary_results: List[BoundaryExtractionResult],
                             training_history: Optional[Dict] = None) -> None:
        """
        Add boundary data for an architecture.
        
        Parameters:
        - label: Architecture label for identification
        - boundary_results: List of BoundaryExtractionResult objects
        - training_history: Optional training history data
        """
        self.architecture_data[label] = {
            'boundary_results': boundary_results,
            'training_history': training_history or {}
        }
        print(f"Added architecture '{label}' with {len(boundary_results)} boundary results")
    
    def load_architecture_from_file(self, label: str, results_path: str) -> None:
        """
        Load architecture data from file.
        
        Parameters:
        - label: Architecture label
        - results_path: Path to results file
        """
        try:
            if results_path.endswith('.pt'):
                data = torch.load(results_path, map_location='cpu')
                boundary_results = data.get('boundary_results', [])
                training_history = data.get('training_history', {})
                self.add_architecture_data(label, boundary_results, training_history)
            else:
                print(f"Unsupported file format: {results_path}")
        except Exception as e:
            print(f"Error loading architecture data: {e}")
    
    def compute_topology_stability(self, boundary_results: List[BoundaryExtractionResult],
                                  window_size: int = 5) -> float:
        """
        Compute topology stability metric.
        
        Parameters:
        - boundary_results: List of boundary results
        - window_size: Window size for stability computation
        
        Returns:
        - stability: Stability metric (higher = more stable)
        """
        if len(boundary_results) < window_size:
            return 0.0
        
        # Sort by epoch
        sorted_results = sorted(boundary_results, key=lambda x: x.epoch)
        
        # Extract Betti numbers over time
        betti_sequences = {}
        for result in sorted_results:
            if result.betti_numbers is not None:
                for i, betti in enumerate(result.betti_numbers):
                    if i not in betti_sequences:
                        betti_sequences[i] = []
                    betti_sequences[i].append(betti)
        
        if not betti_sequences:
            return 0.0
        
        # Compute stability for each Betti dimension
        stabilities = []
        for dim, sequence in betti_sequences.items():
            if len(sequence) >= window_size:
                # Compute variance in sliding windows
                variances = []
                for i in range(len(sequence) - window_size + 1):
                    window = sequence[i:i + window_size]
                    variances.append(np.var(window))
                
                # Stability is inverse of mean variance
                mean_variance = np.mean(variances)
                stability = 1.0 / (1.0 + mean_variance)
                stabilities.append(stability)
        
        return np.mean(stabilities) if stabilities else 0.0
    
    def find_convergence_epoch(self, boundary_results: List[BoundaryExtractionResult],
                              threshold: float = 0.1, window_size: int = 10) -> Optional[int]:
        """
        Find the epoch where topology converges.
        
        Parameters:
        - boundary_results: List of boundary results
        - threshold: Variance threshold for convergence
        - window_size: Window size for convergence detection
        
        Returns:
        - convergence_epoch: Epoch where topology converges (None if not found)
        """
        if len(boundary_results) < window_size:
            return None
        
        # Sort by epoch
        sorted_results = sorted(boundary_results, key=lambda x: x.epoch)
        
        # Extract total topology complexity over time
        complexity_sequence = []
        epochs = []
        
        for result in sorted_results:
            if result.betti_numbers is not None:
                complexity = sum(result.betti_numbers)
                complexity_sequence.append(complexity)
                epochs.append(result.epoch)
        
        if len(complexity_sequence) < window_size:
            return None
        
        # Find convergence point
        for i in range(window_size, len(complexity_sequence)):
            window = complexity_sequence[i-window_size:i]
            if np.var(window) <= threshold:
                return epochs[i-window_size]
        
        return None
    
    def compute_topology_complexity(self, boundary_results: List[BoundaryExtractionResult]) -> float:
        """
        Compute average topology complexity.
        
        Parameters:
        - boundary_results: List of boundary results
        
        Returns:
        - complexity: Average topology complexity
        """
        complexities = []
        for result in boundary_results:
            if result.betti_numbers is not None:
                complexity = sum(result.betti_numbers)
                complexities.append(complexity)
        
        return np.mean(complexities) if complexities else 0.0
    
    def compute_shape_variance(self, boundary_results: List[BoundaryExtractionResult]) -> float:
        """
        Compute variance in boundary shape over training.
        
        Parameters:
        - boundary_results: List of boundary results
        
        Returns:
        - variance: Shape variance metric
        """
        # Compute based on the number of boundary points (proxy for shape complexity)
        point_counts = []
        for result in boundary_results:
            if result.boundary_points is not None:
                point_counts.append(len(result.boundary_points))
        
        return np.var(point_counts) if len(point_counts) > 1 else 0.0
    
    def compute_architecture_metrics(self, label: str) -> TopologyMetrics:
        """
        Compute comprehensive metrics for an architecture.
        
        Parameters:
        - label: Architecture label
        
        Returns:
        - metrics: TopologyMetrics object
        """
        if label not in self.architecture_data:
            raise ValueError(f"Architecture '{label}' not found")
        
        boundary_results = self.architecture_data[label]['boundary_results']
        
        # Compute individual metrics
        stability = self.compute_topology_stability(boundary_results)
        convergence_epoch = self.find_convergence_epoch(boundary_results)
        complexity = self.compute_topology_complexity(boundary_results)
        shape_variance = self.compute_shape_variance(boundary_results)
        
        # Get final topology
        final_result = max(boundary_results, key=lambda x: x.epoch)
        final_topology = final_result.betti_numbers if final_result.betti_numbers else []
        
        return TopologyMetrics(
            betti_stability=stability,
            convergence_epoch=convergence_epoch,
            final_topology=final_topology,
            topology_complexity=complexity,
            shape_variance=shape_variance,
            architecture_label=label
        )
    
    def compute_topology_distance(self, results1: List[BoundaryExtractionResult],
                                 results2: List[BoundaryExtractionResult]) -> float:
        """
        Compute distance between topology trajectories of two architectures.
        
        Parameters:
        - results1: Boundary results for first architecture
        - results2: Boundary results for second architecture
        
        Returns:
        - distance: Topology trajectory distance
        """
        # Extract Betti sequences for both architectures
        def extract_betti_sequence(results):
            betti_sequence = []
            for result in sorted(results, key=lambda x: x.epoch):
                if result.betti_numbers is not None:
                    betti_sequence.append(result.betti_numbers)
            return betti_sequence
        
        seq1 = extract_betti_sequence(results1)
        seq2 = extract_betti_sequence(results2)
        
        if not seq1 or not seq2:
            return float('inf')
        
        # Align sequences to same length
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
        
        # Compute distances between corresponding Betti vectors
        distances = []
        for b1, b2 in zip(seq1, seq2):
            # Ensure same dimension
            max_dim = max(len(b1), len(b2))
            b1_padded = b1 + [0] * (max_dim - len(b1))
            b2_padded = b2 + [0] * (max_dim - len(b2))
            
            # Euclidean distance
            dist = np.linalg.norm(np.array(b1_padded) - np.array(b2_padded))
            distances.append(dist)
        
        return np.mean(distances)
    
    def perform_statistical_tests(self) -> Dict[str, Any]:
        """
        Perform statistical tests comparing architectures.
        
        Returns:
        - test_results: Dictionary of statistical test results
        """
        if len(self.architecture_data) < 2:
            return {}
        
        # Compute metrics for all architectures
        metrics = {}
        for label in self.architecture_data:
            arch_metrics = self.compute_architecture_metrics(label)
            metrics[label] = arch_metrics
        
        # Prepare data for statistical tests
        stability_values = [m.betti_stability for m in metrics.values()]
        complexity_values = [m.topology_complexity for m in metrics.values()]
        variance_values = [m.shape_variance for m in metrics.values()]
        
        test_results = {}
        
        # ANOVA tests if more than 2 architectures
        if len(self.architecture_data) > 2:
            try:
                stability_groups = [[m.betti_stability] for m in metrics.values()]
                complexity_groups = [[m.topology_complexity] for m in metrics.values()]
                
                stability_anova = stats.kruskal(*stability_groups)
                complexity_anova = stats.kruskal(*complexity_groups)
                
                test_results['stability_anova'] = {
                    'statistic': stability_anova.statistic,
                    'p_value': stability_anova.pvalue
                }
                test_results['complexity_anova'] = {
                    'statistic': complexity_anova.statistic,
                    'p_value': complexity_anova.pvalue
                }
            except Exception as e:
                print(f"Error in ANOVA tests: {e}")
        
        # Pairwise comparisons
        labels = list(self.architecture_data.keys())
        pairwise_tests = {}
        
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                label1, label2 = labels[i], labels[j]
                
                # Get data for comparison
                data1 = self.architecture_data[label1]['boundary_results']
                data2 = self.architecture_data[label2]['boundary_results']
                
                # Extract Betti number sequences
                betti1 = [sum(r.betti_numbers) for r in data1 if r.betti_numbers]
                betti2 = [sum(r.betti_numbers) for r in data2 if r.betti_numbers]
                
                if betti1 and betti2:
                    try:
                        # Mann-Whitney U test (non-parametric)
                        u_stat, u_p = stats.mannwhitneyu(betti1, betti2, alternative='two-sided')
                        
                        pairwise_tests[f'{label1}_vs_{label2}'] = {
                            'mann_whitney': {
                                'statistic': u_stat,
                                'p_value': u_p
                            },
                            'effect_size': abs(np.mean(betti1) - np.mean(betti2)) / np.sqrt((np.var(betti1) + np.var(betti2)) / 2)
                        }
                    except Exception as e:
                        print(f"Error in pairwise test {label1} vs {label2}: {e}")
        
        test_results['pairwise_comparisons'] = pairwise_tests
        
        return test_results
    
    def create_similarity_matrix(self) -> np.ndarray:
        """
        Create similarity matrix between architectures.
        
        Returns:
        - similarity_matrix: Pairwise similarity matrix
        """
        labels = list(self.architecture_data.keys())
        n_archs = len(labels)
        
        if n_archs < 2:
            return np.array([[1.0]])
        
        # Compute pairwise distances
        distance_matrix = np.zeros((n_archs, n_archs))
        
        for i in range(n_archs):
            for j in range(n_archs):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    results1 = self.architecture_data[labels[i]]['boundary_results']
                    results2 = self.architecture_data[labels[j]]['boundary_results']
                    distance = self.compute_topology_distance(results1, results2)
                    distance_matrix[i, j] = distance
        
        # Convert distances to similarities
        max_distance = np.max(distance_matrix)
        if max_distance > 0:
            similarity_matrix = 1.0 - (distance_matrix / max_distance)
        else:
            similarity_matrix = np.ones_like(distance_matrix)
        
        return similarity_matrix
    
    def compare_architectures(self) -> ComparisonResult:
        """
        Perform comprehensive comparison of all architectures.
        
        Returns:
        - comparison_result: ComparisonResult object with all analysis
        """
        if len(self.architecture_data) < 2:
            raise ValueError("Need at least 2 architectures for comparison")
        
        print("Performing architecture comparison...")
        
        # Compute metrics for all architectures
        architecture_labels = list(self.architecture_data.keys())
        topology_metrics = []
        
        for label in architecture_labels:
            metrics = self.compute_architecture_metrics(label)
            topology_metrics.append(metrics)
        
        # Perform statistical tests
        statistical_tests = self.perform_statistical_tests()
        
        # Create similarity matrix
        similarity_matrix = self.create_similarity_matrix()
        
        # Convergence comparison
        convergence_comparison = {
            'convergence_epochs': {m.architecture_label: m.convergence_epoch for m in topology_metrics},
            'mean_convergence': np.mean([m.convergence_epoch for m in topology_metrics if m.convergence_epoch is not None])
        }
        
        # Stability comparison
        stability_comparison = {
            'stability_scores': {m.architecture_label: m.betti_stability for m in topology_metrics},
            'stability_ranking': sorted(topology_metrics, key=lambda x: x.betti_stability, reverse=True)
        }
        
        self.comparison_results = ComparisonResult(
            architecture_labels=architecture_labels,
            topology_metrics=topology_metrics,
            statistical_tests=statistical_tests,
            similarity_matrix=similarity_matrix,
            convergence_comparison=convergence_comparison,
            stability_comparison=stability_comparison
        )
        
        print(f"Comparison completed for {len(architecture_labels)} architectures")
        
        return self.comparison_results
    
    def create_comparison_plots(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive comparison plots.
        
        Parameters:
        - output_dir: Directory to save plots (optional)
        
        Returns:
        - plots: Dictionary of plot objects
        """
        if self.comparison_results is None:
            self.compare_architectures()
        
        plots = {}
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Stability comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = self.comparison_results.architecture_labels
        stabilities = [m.betti_stability for m in self.comparison_results.topology_metrics]
        
        bars = ax.bar(labels, stabilities, alpha=0.7)
        ax.set_ylabel('Topology Stability')
        ax.set_title('Architecture Stability Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, stability in zip(bars, stabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{stability:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['stability_comparison'] = fig
        
        if output_dir:
            plt.savefig(output_path / 'stability_comparison.png', dpi=300, bbox_inches='tight')
        
        # 2. Complexity comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        complexities = [m.topology_complexity for m in self.comparison_results.topology_metrics]
        
        bars = ax.bar(labels, complexities, alpha=0.7, color='orange')
        ax.set_ylabel('Average Topology Complexity')
        ax.set_title('Architecture Complexity Comparison')
        
        for bar, complexity in zip(bars, complexities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{complexity:.2f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['complexity_comparison'] = fig
        
        if output_dir:
            plt.savefig(output_path / 'complexity_comparison.png', dpi=300, bbox_inches='tight')
        
        # 3. Similarity heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(self.comparison_results.similarity_matrix,
                   xticklabels=labels,
                   yticklabels=labels,
                   annot=True,
                   cmap='viridis',
                   vmin=0, vmax=1,
                   ax=ax)
        
        ax.set_title('Architecture Topology Similarity Matrix')
        plt.tight_layout()
        plots['similarity_heatmap'] = fig
        
        if output_dir:
            plt.savefig(output_path / 'similarity_heatmap.png', dpi=300, bbox_inches='tight')
        
        # 4. Convergence comparison
        convergence_epochs = [m.convergence_epoch for m in self.comparison_results.topology_metrics]
        valid_convergence = [(label, epoch) for label, epoch in zip(labels, convergence_epochs) if epoch is not None]
        
        if valid_convergence:
            fig, ax = plt.subplots(figsize=(10, 6))
            conv_labels, conv_epochs = zip(*valid_convergence)
            
            bars = ax.bar(conv_labels, conv_epochs, alpha=0.7, color='green')
            ax.set_ylabel('Convergence Epoch')
            ax.set_title('Topology Convergence Comparison')
            
            for bar, epoch in zip(bars, conv_epochs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{epoch}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plots['convergence_comparison'] = fig
            
            if output_dir:
                plt.savefig(output_path / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
        
        return plots
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive analysis report.
        
        Parameters:
        - output_path: Path to save report (optional)
        
        Returns:
        - report: Analysis report as string
        """
        if self.comparison_results is None:
            self.compare_architectures()
        
        report_lines = []
        report_lines.append("DECISION BOUNDARY TOPOLOGY ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Architectures analyzed: {len(self.comparison_results.architecture_labels)}")
        report_lines.append("")
        
        # Individual architecture metrics
        report_lines.append("INDIVIDUAL ARCHITECTURE METRICS")
        report_lines.append("-" * 35)
        
        for metrics in self.comparison_results.topology_metrics:
            report_lines.append(f"\nArchitecture: {metrics.architecture_label}")
            report_lines.append(f"  Topology Stability: {metrics.betti_stability:.4f}")
            report_lines.append(f"  Convergence Epoch: {metrics.convergence_epoch or 'Not found'}")
            report_lines.append(f"  Average Complexity: {metrics.topology_complexity:.4f}")
            report_lines.append(f"  Shape Variance: {metrics.shape_variance:.4f}")
            report_lines.append(f"  Final Topology: {metrics.final_topology}")
        
        # Stability ranking
        report_lines.append("\nSTABILITY RANKING")
        report_lines.append("-" * 17)
        
        stability_ranking = self.comparison_results.stability_comparison['stability_ranking']
        for i, metrics in enumerate(stability_ranking):
            report_lines.append(f"{i+1}. {metrics.architecture_label}: {metrics.betti_stability:.4f}")
        
        # Statistical tests
        if self.comparison_results.statistical_tests:
            report_lines.append("\nSTATISTICAL ANALYSIS")
            report_lines.append("-" * 20)
            
            stats_tests = self.comparison_results.statistical_tests
            
            if 'stability_anova' in stats_tests:
                anova = stats_tests['stability_anova']
                report_lines.append(f"Stability ANOVA: H={anova['statistic']:.4f}, p={anova['p_value']:.4f}")
            
            if 'complexity_anova' in stats_tests:
                anova = stats_tests['complexity_anova']
                report_lines.append(f"Complexity ANOVA: H={anova['statistic']:.4f}, p={anova['p_value']:.4f}")
            
            if 'pairwise_comparisons' in stats_tests:
                report_lines.append("\nPairwise Comparisons:")
                for comparison, results in stats_tests['pairwise_comparisons'].items():
                    mw = results['mann_whitney']
                    effect = results['effect_size']
                    report_lines.append(f"  {comparison}: U={mw['statistic']:.2f}, p={mw['p_value']:.4f}, d={effect:.4f}")
        
        # Convergence analysis
        report_lines.append("\nCONVERGENCE ANALYSIS")
        report_lines.append("-" * 20)
        
        conv_comparison = self.comparison_results.convergence_comparison
        for arch, epoch in conv_comparison['convergence_epochs'].items():
            report_lines.append(f"{arch}: {epoch or 'Not converged'}")
        
        if conv_comparison['mean_convergence']:
            report_lines.append(f"Mean convergence epoch: {conv_comparison['mean_convergence']:.1f}")
        
        # Summary and recommendations
        report_lines.append("\nSUMMARY AND RECOMMENDATIONS")
        report_lines.append("-" * 30)
        
        # Find best architecture
        best_stability = max(self.comparison_results.topology_metrics, key=lambda x: x.betti_stability)
        fastest_convergence = min([m for m in self.comparison_results.topology_metrics if m.convergence_epoch],
                                 key=lambda x: x.convergence_epoch, default=None)
        
        report_lines.append(f"Most stable architecture: {best_stability.architecture_label} (stability: {best_stability.betti_stability:.4f})")
        
        if fastest_convergence:
            report_lines.append(f"Fastest convergence: {fastest_convergence.architecture_label} (epoch: {fastest_convergence.convergence_epoch})")
        
        # Generate recommendations
        report_lines.append("\nRecommendations:")
        if best_stability.betti_stability > 0.8:
            report_lines.append("- High stability achieved, architecture is well-suited for the task")
        elif best_stability.betti_stability < 0.5:
            report_lines.append("- Low stability detected, consider architectural modifications")
        
        if fastest_convergence and fastest_convergence.convergence_epoch < 50:
            report_lines.append("- Fast convergence observed, architecture learns efficiently")
        
        report = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved: {output_path}")
        
        return report
    
    def export_results(self, output_path: str) -> None:
        """
        Export analysis results to file.
        
        Parameters:
        - output_path: Path to save results
        """
        if self.comparison_results is None:
            self.compare_architectures()
        
        try:
            export_data = {
                'comparison_results': self.comparison_results,
                'architecture_data': self.architecture_data,
                'config': self.config
            }
            
            if output_path.endswith('.pt'):
                torch.save(export_data, output_path)
            elif output_path.endswith('.csv'):
                # Export as CSV for spreadsheet analysis
                metrics_data = []
                for metrics in self.comparison_results.topology_metrics:
                    metrics_data.append({
                        'architecture': metrics.architecture_label,
                        'stability': metrics.betti_stability,
                        'convergence_epoch': metrics.convergence_epoch,
                        'complexity': metrics.topology_complexity,
                        'shape_variance': metrics.shape_variance,
                        'final_betti_0': metrics.final_topology[0] if len(metrics.final_topology) > 0 else 0,
                        'final_betti_1': metrics.final_topology[1] if len(metrics.final_topology) > 1 else 0,
                        'final_betti_2': metrics.final_topology[2] if len(metrics.final_topology) > 2 else 0,
                    })
                
                df = pd.DataFrame(metrics_data)
                df.to_csv(output_path, index=False)
            
            print(f"Results exported: {output_path}")
            
        except Exception as e:
            print(f"Error exporting results: {e}")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Boundary Topology Comparison Analysis")
    parser.add_argument('--config', type=str, help='Path to analysis config file')
    parser.add_argument('--results', type=str, nargs='+', help='Paths to architecture result files')
    parser.add_argument('--labels', type=str, nargs='+', help='Labels for architectures')
    parser.add_argument('--output-dir', type=str, default='results/topology_analysis', help='Output directory')
    parser.add_argument('--report', action='store_true', help='Generate analysis report')
    parser.add_argument('--plots', action='store_true', help='Generate comparison plots')
    parser.add_argument('--export', type=str, help='Export results to file')
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = {}
        if args.config:
            config = load_boundary_config(args.config)
        
        # Create analyzer
        analyzer = BoundaryTopologyAnalyzer(config)
        
        # Load architecture data
        if args.results:
            labels = args.labels or [f'Architecture_{i+1}' for i in range(len(args.results))]
            
            for label, results_path in zip(labels, args.results):
                analyzer.load_architecture_from_file(label, results_path)
        
        if len(analyzer.architecture_data) < 2:
            print("Error: Need at least 2 architectures for comparison")
            return
        
        # Perform analysis
        print("Performing comprehensive topology analysis...")
        comparison_results = analyzer.compare_architectures()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        if args.report:
            report_path = output_dir / 'topology_analysis_report.txt'
            report = analyzer.generate_report(str(report_path))
            print(f"\nAnalysis report saved: {report_path}")
        
        # Generate plots
        if args.plots:
            plots_dir = output_dir / 'plots'
            plots = analyzer.create_comparison_plots(str(plots_dir))
            print(f"Comparison plots saved to: {plots_dir}")
        
        # Export results
        if args.export:
            export_path = output_dir / args.export
            analyzer.export_results(str(export_path))
        
        print("\nTopology analysis completed successfully!")
        
        # Print summary
        print(f"\nSummary:")
        print(f"Architectures analyzed: {len(comparison_results.architecture_labels)}")
        
        best_stability = max(comparison_results.topology_metrics, key=lambda x: x.betti_stability)
        print(f"Most stable: {best_stability.architecture_label} (stability: {best_stability.betti_stability:.4f})")
        
    except Exception as e:
        print(f"Error in topology analysis: {e}")
        raise


if __name__ == "__main__":
    main()