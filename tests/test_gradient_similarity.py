"""
Tests for gradient-based similarity analysis module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.gradient_similarity import (
    GradientSimilarityAnalyzer, 
    GradientFlowSnapshot,
    LossLandscapePoint
)


class SimpleNet(nn.Module):
    """Simple test network."""
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


@pytest.fixture
def device():
    """Get test device."""
    return torch.device('cpu')


@pytest.fixture
def analyzer(device):
    """Create analyzer instance."""
    return GradientSimilarityAnalyzer(device=device)


@pytest.fixture
def simple_data(device):
    """Create simple test dataset."""
    torch.manual_seed(42)
    X = torch.randn(100, 2).to(device)
    y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1).to(device)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    return loader


@pytest.fixture
def simple_models(device):
    """Create test models."""
    torch.manual_seed(42)
    model1 = SimpleNet().to(device)
    model2 = SimpleNet().to(device)
    
    # Make models slightly different
    with torch.no_grad():
        for p in model2.parameters():
            p.data += torch.randn_like(p) * 0.1
    
    return model1, model2


class TestGradientFlowAnalysis:
    """Test gradient flow tracking and analysis."""
    
    def test_track_gradient_flow(self, analyzer, simple_models, simple_data):
        """Test gradient flow tracking."""
        model = simple_models[0]
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        flow = analyzer.track_gradient_flow(
            model, loss_fn, simple_data, optimizer, num_steps=5
        )
        
        assert len(flow) == 5
        assert all(isinstance(s, GradientFlowSnapshot) for s in flow)
        assert all(s.step == i for i, s in enumerate(flow))
        
        # Check that loss generally decreases
        losses = [s.loss for s in flow]
        assert losses[-1] < losses[0]
    
    def test_gradient_flow_similarity(self, analyzer, simple_models, simple_data):
        """Test gradient flow similarity computation."""
        model1, model2 = simple_models
        loss_fn = nn.BCELoss()
        
        # Track flows
        opt1 = torch.optim.Adam(model1.parameters(), lr=0.01)
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        
        flow1 = analyzer.track_gradient_flow(model1, loss_fn, simple_data, opt1, num_steps=5)
        flow2 = analyzer.track_gradient_flow(model2, loss_fn, simple_data, opt2, num_steps=5)
        
        # Test different similarity methods
        methods = ['trajectory', 'velocity', 'curvature']
        for method in methods:
            sim = analyzer.compute_gradient_flow_similarity(flow1, flow2, method=method)
            assert 0 <= sim <= 1 or method == 'trajectory'  # trajectory uses distance
    
    def test_critical_points_analysis(self, analyzer, simple_models, simple_data):
        """Test critical point analysis."""
        model = simple_models[0]
        loss_fn = nn.BCELoss()
        
        critical = analyzer.analyze_critical_points(
            model, loss_fn, simple_data, num_perturbations=10
        )
        
        assert 'num_minima' in critical
        assert 'num_saddles' in critical
        assert 'num_maxima' in critical
        assert 'average_morse_index' in critical
        assert critical['num_minima'] + critical['num_saddles'] + critical['num_maxima'] == 10


class TestLossLandscapeAnalysis:
    """Test loss landscape analysis."""
    
    def test_analyze_loss_landscape(self, analyzer, simple_models, simple_data):
        """Test loss landscape analysis."""
        model = simple_models[0]
        loss_fn = nn.BCELoss()
        
        landscape = analyzer.analyze_loss_landscape(
            model, loss_fn, simple_data,
            resolution=10, epsilon=0.1
        )
        
        assert 'loss_surface' in landscape
        assert landscape['loss_surface'].shape == (10, 10)
        assert 'roughness' in landscape
        assert 'convexity' in landscape
        assert 'barrier_heights' in landscape
        assert 'basin_volume' in landscape
        
        # Check reasonable values
        assert landscape['roughness'] >= 0
        assert 0 <= landscape['convexity'] <= 1
        assert landscape['basin_volume'] >= 0
    
    def test_compare_loss_landscapes(self, analyzer, simple_models, simple_data):
        """Test loss landscape comparison."""
        model1, model2 = simple_models
        loss_fn = nn.BCELoss()
        
        landscape1 = analyzer.analyze_loss_landscape(
            model1, loss_fn, simple_data, resolution=10
        )
        landscape2 = analyzer.analyze_loss_landscape(
            model2, loss_fn, simple_data, resolution=10
        )
        
        comparison = analyzer.compare_loss_landscapes(landscape1, landscape2)
        
        assert 'surface_correlation' in comparison
        assert 'roughness_ratio' in comparison
        assert 'convexity_difference' in comparison
        assert 'basin_volume_ratio' in comparison
        
        # Check correlation is valid
        assert -1 <= comparison['surface_correlation'] <= 1


class TestHessianAnalysis:
    """Test Hessian-based analysis."""
    
    def test_compute_hessian_similarity(self, analyzer, simple_models, simple_data):
        """Test Hessian similarity computation."""
        model1, model2 = simple_models
        loss_fn = nn.BCELoss()
        
        methods = ['eigenvalue', 'trace', 'determinant', 'condition']
        for method in methods:
            sim = analyzer.compute_hessian_similarity(
                model1, model2, loss_fn, simple_data,
                method=method, top_k=5
            )
            assert 0 <= sim <= 1
    
    def test_analyze_curvature(self, analyzer, simple_models, simple_data):
        """Test curvature analysis."""
        model = simple_models[0]
        loss_fn = nn.BCELoss()
        
        curvature = analyzer.analyze_curvature(
            model, loss_fn, simple_data,
            num_directions=5
        )
        
        assert 'mean_curvature' in curvature
        assert 'max_curvature' in curvature
        assert 'min_curvature' in curvature
        assert 'curvature_variance' in curvature
        assert 'negative_curvature_ratio' in curvature
        assert len(curvature['curvature_distribution']) == 5


class TestNTKAnalysis:
    """Test Neural Tangent Kernel analysis."""
    
    def test_compute_ntk(self, analyzer, simple_models, device):
        """Test NTK computation."""
        model = simple_models[0]
        x1 = torch.randn(5, 2).to(device)
        x2 = torch.randn(3, 2).to(device)
        
        K = analyzer.compute_ntk(model, x1, x2)
        
        assert K.shape == (5, 3)
        assert torch.isfinite(K).all()
    
    def test_compare_ntk_similarity(self, analyzer, simple_models, simple_data):
        """Test NTK similarity comparison."""
        model1, model2 = simple_models
        
        similarities = analyzer.compare_ntk_similarity(
            model1, model2, simple_data,
            num_samples=20
        )
        
        assert 'kernel_alignment' in similarities
        assert 'frobenius_similarity' in similarities
        assert 'eigenvalue_similarity' in similarities
        assert 'trace_similarity' in similarities
        
        # Check values are reasonable
        for key, value in similarities.items():
            if isinstance(value, float):
                assert -1 <= value <= 1 or key == 'trace_similarity'


class TestOptimizationDynamics:
    """Test optimization dynamics tracking."""
    
    def test_track_optimization_dynamics(self, analyzer, simple_models, simple_data):
        """Test optimization dynamics tracking."""
        model = simple_models[0]
        loss_fn = nn.BCELoss()
        
        dynamics = analyzer.track_optimization_dynamics(
            model, loss_fn, simple_data,
            optimizer_class=torch.optim.Adam,
            lr=0.01,
            num_epochs=3
        )
        
        expected_keys = ['loss', 'gradient_norm', 'parameter_change', 
                        'effective_lr', 'gradient_variance', 'gradient_cosine']
        
        for key in expected_keys:
            assert key in dynamics
            assert len(dynamics[key]) <= 3  # At most num_epochs entries
    
    def test_compare_optimization_dynamics(self, analyzer, simple_models, simple_data):
        """Test optimization dynamics comparison."""
        model1, model2 = simple_models
        loss_fn = nn.BCELoss()
        
        # Track dynamics for both models
        dynamics1 = analyzer.track_optimization_dynamics(
            model1, loss_fn, simple_data,
            optimizer_class=torch.optim.Adam,
            lr=0.01,
            num_epochs=2
        )
        dynamics2 = analyzer.track_optimization_dynamics(
            model2, loss_fn, simple_data,
            optimizer_class=torch.optim.Adam,
            lr=0.01,
            num_epochs=2
        )
        
        comparison = analyzer.compare_optimization_dynamics(dynamics1, dynamics2)
        
        assert 'overall_dynamics_similarity' in comparison
        assert -1 <= comparison['overall_dynamics_similarity'] <= 1


class TestHelperMethods:
    """Test helper methods."""
    
    def test_get_set_flat_params(self, analyzer, simple_models):
        """Test parameter flattening and setting."""
        model = simple_models[0]
        
        # Get parameters
        params = analyzer._get_flat_params(model)
        assert params.dim() == 1
        assert params.numel() == sum(p.numel() for p in model.parameters())
        
        # Modify and set
        new_params = params + 0.1
        analyzer._set_flat_params(model, new_params)
        
        # Check modification
        params_after = analyzer._get_flat_params(model)
        assert torch.allclose(params_after, new_params)
    
    def test_compute_directional_curvature(self, analyzer, simple_models, simple_data):
        """Test directional curvature computation."""
        model = simple_models[0]
        loss_fn = nn.BCELoss()
        
        # Random direction
        d = sum(p.numel() for p in model.parameters())
        direction = torch.randn(d).to(model.fc1.weight.device)
        direction = direction / direction.norm()
        
        curvature = analyzer._compute_directional_curvature(
            model, loss_fn, simple_data, direction
        )
        
        assert isinstance(curvature, float)
        assert np.isfinite(curvature)


class TestVisualization:
    """Test visualization methods."""
    
    def test_visualize_gradient_flow(self, analyzer, simple_models, simple_data, tmp_path):
        """Test gradient flow visualization."""
        model = simple_models[0]
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        flow = analyzer.track_gradient_flow(
            model, loss_fn, simple_data, optimizer, num_steps=5
        )
        
        save_path = tmp_path / "gradient_flow.png"
        analyzer.visualize_gradient_flow(flow, save_path=str(save_path))
        
        assert save_path.exists()
    
    def test_visualize_loss_landscape(self, analyzer, simple_models, simple_data, tmp_path):
        """Test loss landscape visualization."""
        model = simple_models[0]
        loss_fn = nn.BCELoss()
        
        landscape = analyzer.analyze_loss_landscape(
            model, loss_fn, simple_data, resolution=10
        )
        
        save_path = tmp_path / "loss_landscape.png"
        analyzer.visualize_loss_landscape(landscape, save_path=str(save_path))
        
        assert save_path.exists()


class TestIntegration:
    """Integration tests."""
    
    def test_full_similarity_analysis(self, analyzer, simple_models, simple_data):
        """Test complete similarity analysis pipeline."""
        model1, model2 = simple_models
        loss_fn = nn.BCELoss()
        
        # Track gradient flows
        opt1 = torch.optim.Adam(model1.parameters(), lr=0.01)
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        
        flow1 = analyzer.track_gradient_flow(model1, loss_fn, simple_data, opt1, num_steps=3)
        flow2 = analyzer.track_gradient_flow(model2, loss_fn, simple_data, opt2, num_steps=3)
        
        # Compute various similarities
        flow_sim = analyzer.compute_gradient_flow_similarity(flow1, flow2, method='velocity')
        hessian_sim = analyzer.compute_hessian_similarity(
            model1, model2, loss_fn, simple_data, method='eigenvalue', top_k=5
        )
        ntk_sim = analyzer.compare_ntk_similarity(
            model1, model2, simple_data, num_samples=20
        )
        
        # Combined similarity
        overall_sim = (flow_sim + hessian_sim + ntk_sim['kernel_alignment']) / 3
        
        assert 0 <= overall_sim <= 1
        
        # Models should be somewhat similar (same architecture)
        assert overall_sim > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])