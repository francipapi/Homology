"""
Network Graph Builder Module

This module provides classes to construct graph representations of neural networks
for topological analysis. It implements the factor graph approach for convolutional
layers to handle weight sharing efficiently.

Key Components:
- NetworkGraphBuilder: Base class for graph construction
- MLPGraphBuilder: Handles fully connected layers with direct edges
- ConvGraphBuilder: Implements factor graph approach for convolutional layers
- UnifiedGraphBuilder: Combines different layer types into a single graph

The factor graph approach for convolutional layers introduces parameter nodes
to represent shared weights, creating a tripartite graph structure:
Input Activations → Parameter Nodes → Output Activations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import graph_tool as gt
from graph_tool import Graph
import scipy.sparse as sp
from abc import ABC, abstractmethod


class NetworkGraphBuilder(ABC):
    """
    Abstract base class for building graph representations of neural networks.
    
    This class provides the interface and common functionality for converting
    neural network architectures into graph structures suitable for topological analysis.
    """
    
    def __init__(self, normalize_weights: bool = True, 
                 weight_threshold: float = 1e-6,
                 handle_negative_weights: bool = True,
                 weight_encoding: str = 'standard',
                 min_edge_distance: float = 1e-6,
                 normalization_type: str = 'standard'):
        """
        Initialize the graph builder.
        
        Args:
            normalize_weights: Whether to normalize edge weights
            weight_threshold: Minimum absolute weight value to include
            handle_negative_weights: Whether to reverse edges for negative weights (deprecated, use weight_encoding)
            weight_encoding: Method for encoding weights ('standard', 'mirror', 'reverse')
            min_edge_distance: Minimum edge distance for reverse encoding
            normalization_type: Type of normalization ('standard' or 'nn_similarity')
        """
        self.normalize_weights = normalize_weights
        self.weight_threshold = weight_threshold
        self.handle_negative_weights = handle_negative_weights
        self.weight_encoding = weight_encoding
        self.min_edge_distance = min_edge_distance
        self.normalization_type = normalization_type
        
        # For nn-similarity style normalization
        self.max_abs_weight = None  # Will be computed when building graph
        
    @abstractmethod
    def build_graph(self, layer: nn.Module, input_shape: Tuple[int, ...], 
                   prev_vertices: Optional[List[int]] = None) -> Tuple[Graph, List[int]]:
        """
        Build graph representation for a neural network layer.
        
        Args:
            layer: The neural network layer to convert
            input_shape: Shape of input to this layer
            prev_vertices: Vertex indices from previous layer (if any)
            
        Returns:
            Tuple of (graph, output_vertices)
        """
        pass
    
    def _create_graph_with_properties(self) -> Tuple[Graph, Any, Any, Any]:
        """Create a new directed graph with standard properties."""
        g = Graph(directed=True)
        
        # Vertex properties
        v_type = g.new_vertex_property("string")  # Type: 'input', 'hidden', 'output', 'parameter'
        v_layer = g.new_vertex_property("int")    # Layer index
        v_neuron_idx = g.new_vertex_property("int")  # Neuron index within layer
        
        # Edge property for weights
        e_weight = g.new_edge_property("double")
        
        # Set as internal properties
        g.vertex_properties["type"] = v_type
        g.vertex_properties["layer"] = v_layer
        g.vertex_properties["neuron_idx"] = v_neuron_idx
        g.edge_properties["weight"] = e_weight
        
        return g, v_type, v_layer, e_weight
    
    def _add_edge_with_sign(self, g: Graph, u: int, v: int, weight: float, 
                           e_weight: Any) -> None:
        """
        Add edge respecting sign convention based on weight_encoding method.
        
        Args:
            g: Graph object
            u: Source vertex
            v: Target vertex  
            weight: Edge weight (can be negative)
            e_weight: Edge weight property map
        """
        if self.weight_encoding == 'standard':
            # Standard encoding: keep original direction and weight
            e = g.add_edge(u, v)
            e_weight[e] = weight  # Keep original sign
        elif self.weight_encoding == 'reverse':
            # Reverse encoding: negative weights reverse direction
            if weight < 0:
                e = g.add_edge(v, u)
                # Use nn-similarity formula: 1 - |w|/max_weight
                if self.max_abs_weight is not None and self.max_abs_weight > 0:
                    e_weight[e] = max(1 - abs(weight) / self.max_abs_weight, self.min_edge_distance)
                else:
                    e_weight[e] = abs(weight)
            else:
                e = g.add_edge(u, v)
                if self.max_abs_weight is not None and self.max_abs_weight > 0:
                    e_weight[e] = max(1 - abs(weight) / self.max_abs_weight, self.min_edge_distance)
                else:
                    e_weight[e] = weight
        elif self.weight_encoding == 'mirror':
            # Mirror encoding handled separately in MLPGraphBuilder
            # This should not be called directly for mirror encoding
            raise ValueError("Mirror encoding requires special handling in build_graph")
        else:
            # Fallback to legacy behavior
            if self.handle_negative_weights and weight < 0:
                e = g.add_edge(v, u)
                e_weight[e] = abs(weight)
            else:
                e = g.add_edge(u, v)
                e_weight[e] = abs(weight)
    
    def _normalize_edge_weights(self, g: Graph, e_weight: Any, 
                               normalization_type: str = 'standard') -> None:
        """
        Normalize edge weights.
        
        Args:
            g: Graph object
            e_weight: Edge weight property map
            normalization_type: 'standard' (divide by max) or 'nn_similarity' (distance-based)
        """
        if not self.normalize_weights:
            return
            
        weights = e_weight.get_array()
        if len(weights) == 0:
            return
        
        if normalization_type == 'nn_similarity':
            # nn-similarity normalization: convert to distances
            # distance = min_edge_distance + max(|max_weight|, |min_weight|) - |weight|
            max_abs = np.max(np.abs(weights))
            if max_abs > 0:
                # Convert weights to distances
                e_weight.a = self.min_edge_distance + max_abs - np.abs(weights)
        else:
            # Standard normalization: divide by max
            max_weight = np.max(weights)
            if max_weight > 0:
                e_weight.a = weights / max_weight


class MLPGraphBuilder(NetworkGraphBuilder):
    """
    Graph builder for fully connected (MLP) layers.
    
    Creates direct edges between neurons in consecutive layers,
    with edge weights corresponding to the connection strengths.
    """
    
    def build_graph(self, layer: nn.Linear, input_shape: Tuple[int, ...], 
                   prev_vertices: Optional[List[int]] = None) -> Tuple[Graph, List[int]]:
        """
        Build graph for a fully connected layer.
        
        Args:
            layer: Linear layer
            input_shape: Input shape (should be 1D for Linear layers)
            prev_vertices: Vertices from previous layer
            
        Returns:
            Tuple of (graph, output_vertices)
        """
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"Expected nn.Linear, got {type(layer)}")
            
        # Create new graph
        g, v_type, v_layer, e_weight = self._create_graph_with_properties()
        
        # Get layer dimensions
        in_features = layer.in_features
        out_features = layer.out_features
        weights = layer.weight.detach().cpu().numpy()  # Shape: (out_features, in_features)
        
        # For mirror encoding, we need to create mirror vertices
        mirror_suffix = '_m'
        
        # Create or reuse input vertices
        if prev_vertices is None:
            # Create input vertices
            input_vertices = []
            input_vertices_mirror = []
            for i in range(in_features):
                v = g.add_vertex()
                v_type[v] = "input"
                v_layer[v] = 0
                g.vp.neuron_idx[v] = i
                input_vertices.append(v)
                
                # Create mirror vertex for mirror encoding
                if self.weight_encoding == 'mirror':
                    v_m = g.add_vertex()
                    v_type[v_m] = "input" + mirror_suffix
                    v_layer[v_m] = 0
                    g.vp.neuron_idx[v_m] = i
                    input_vertices_mirror.append(v_m)
        else:
            # Use previous layer's vertices
            input_vertices = prev_vertices
            # For mirror encoding, we assume prev_vertices contains both regular and mirror vertices
            if self.weight_encoding == 'mirror':
                # Split vertices into regular and mirror
                half = len(prev_vertices) // 2
                input_vertices = prev_vertices[:half]
                input_vertices_mirror = prev_vertices[half:]
            # Ensure we have the right number of input vertices
            if len(input_vertices) != in_features:
                raise ValueError(f"Mismatch: prev_vertices has {len(input_vertices)} vertices but layer expects {in_features} inputs")
            
        # Create output vertices
        output_vertices = []
        output_vertices_mirror = []
        for i in range(out_features):
            v = g.add_vertex()
            v_type[v] = "hidden"
            v_layer[v] = 1
            g.vp.neuron_idx[v] = i
            output_vertices.append(v)
            
            # Create mirror vertex for mirror encoding
            if self.weight_encoding == 'mirror':
                v_m = g.add_vertex()
                v_type[v_m] = "hidden" + mirror_suffix
                v_layer[v_m] = 1
                g.vp.neuron_idx[v_m] = i
                output_vertices_mirror.append(v_m)
        
        # Add edges with weights
        if self.weight_encoding == 'mirror':
            # Mirror encoding: positive weights use regular vertices, negative use mirror vertices
            for i in range(out_features):
                for j in range(in_features):
                    weight = weights[i, j]
                    
                    # Skip near-zero weights
                    if abs(weight) < self.weight_threshold:
                        continue
                    
                    if weight > 0:
                        # Positive weight: regular vertices
                        e = g.add_edge(input_vertices[j], output_vertices[i])
                        e_weight[e] = abs(weight)
                    else:
                        # Negative weight: mirror vertices
                        e = g.add_edge(input_vertices_mirror[j], output_vertices_mirror[i])
                        e_weight[e] = abs(weight)
        else:
            # Standard or reverse encoding
            for i in range(out_features):
                for j in range(in_features):
                    weight = weights[i, j]
                    
                    # Skip near-zero weights
                    if abs(weight) < self.weight_threshold:
                        continue
                        
                    self._add_edge_with_sign(g, input_vertices[j], output_vertices[i], 
                                           weight, e_weight)
        
        # Handle bias as edges from a special bias vertex
        if layer.bias is not None:
            bias = layer.bias.detach().cpu().numpy()
            bias_vertex = g.add_vertex()
            v_type[bias_vertex] = "bias"
            v_layer[bias_vertex] = 0
            g.vp.neuron_idx[bias_vertex] = -1
            
            if self.weight_encoding == 'mirror':
                # Create mirror bias vertex
                bias_vertex_mirror = g.add_vertex()
                v_type[bias_vertex_mirror] = "bias" + mirror_suffix
                v_layer[bias_vertex_mirror] = 0
                g.vp.neuron_idx[bias_vertex_mirror] = -1
                
                for i in range(out_features):
                    if abs(bias[i]) >= self.weight_threshold:
                        if bias[i] > 0:
                            e = g.add_edge(bias_vertex, output_vertices[i])
                            e_weight[e] = abs(bias[i])
                        else:
                            e = g.add_edge(bias_vertex_mirror, output_vertices_mirror[i])
                            e_weight[e] = abs(bias[i])
            else:
                for i in range(out_features):
                    if abs(bias[i]) >= self.weight_threshold:
                        self._add_edge_with_sign(g, bias_vertex, output_vertices[i], 
                                               bias[i], e_weight)
        
        # Normalize weights if requested
        self._normalize_edge_weights(g, e_weight, self.normalization_type)
        
        # Return vertices (for mirror encoding, return both regular and mirror vertices)
        if self.weight_encoding == 'mirror':
            return g, output_vertices + output_vertices_mirror
        else:
            return g, output_vertices


class ConvGraphBuilder(NetworkGraphBuilder):
    """
    Graph builder for convolutional layers using factor graph approach.
    
    This implementation creates a tripartite graph structure:
    - Input activation nodes: One per spatial position and channel
    - Parameter nodes: One per unique kernel weight (factor nodes)
    - Output activation nodes: One per output spatial position and channel
    
    Weight sharing is handled by having all spatial positions connect through
    the same parameter nodes.
    """
    
    def __init__(self, *args, include_spatial_info: bool = True, **kwargs):
        """
        Initialize ConvGraphBuilder.
        
        Args:
            include_spatial_info: Whether to store spatial position info in vertices
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.include_spatial_info = include_spatial_info
    
    def build_graph(self, layer: Union[nn.Conv1d, nn.Conv2d], 
                   input_shape: Tuple[int, ...],
                   prev_vertices: Optional[List[int]] = None,
                   existing_graph: Optional[Graph] = None) -> Tuple[Graph, List[int]]:
        """
        Build factor graph for a convolutional layer.
        
        Args:
            layer: Convolutional layer (Conv1d or Conv2d)
            input_shape: Input shape (C, H, W) for Conv2d or (C, L) for Conv1d
            prev_vertices: Previous layer vertices (required for middle layers)
            
        Returns:
            Tuple of (graph, output_vertices)
        """
        if isinstance(layer, nn.Conv2d):
            return self._build_conv2d_graph(layer, input_shape, prev_vertices)
        elif isinstance(layer, nn.Conv1d):
            return self._build_conv1d_graph(layer, input_shape, prev_vertices)
        else:
            raise ValueError(f"Expected Conv1d or Conv2d, got {type(layer)}")
    
    def _build_conv2d_graph(self, layer: nn.Conv2d, 
                           input_shape: Tuple[int, int, int],
                           prev_vertices: Optional[List[int]] = None) -> Tuple[Graph, List[int]]:
        """Build factor graph for Conv2d layer."""
        # Extract layer parameters
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        weights = layer.weight.detach().cpu().numpy()  # Shape: (out_c, in_c, k_h, k_w)
        
        # Calculate spatial dimensions
        C_in, H_in, W_in = input_shape
        k_h, k_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
        p_h, p_w = padding if isinstance(padding, tuple) else (padding, padding)
        
        H_out = (H_in + 2 * p_h - k_h) // s_h + 1
        W_out = (W_in + 2 * p_w - k_w) // s_w + 1
        
        # Create graph
        g, v_type, v_layer, e_weight = self._create_graph_with_properties()
        
        # Add spatial position properties if requested
        if self.include_spatial_info:
            v_spatial_h = g.new_vertex_property("int")
            v_spatial_w = g.new_vertex_property("int")
            g.vertex_properties["spatial_h"] = v_spatial_h
            g.vertex_properties["spatial_w"] = v_spatial_w
        
        # Step 1: Create or map input activation vertices
        input_vertices = {}  # (h, w, c) -> vertex
        
        if prev_vertices is None:
            # First conv layer - create input vertices
            for h in range(H_in):
                for w in range(W_in):
                    for c in range(C_in):
                        v = g.add_vertex()
                        v_type[v] = "input"
                        v_layer[v] = 0
                        g.vp.neuron_idx[v] = c + C_in * (w + W_in * h)
                        if self.include_spatial_info:
                            g.vp.spatial_h[v] = h
                            g.vp.spatial_w[v] = w
                        input_vertices[(h, w, c)] = v
        else:
            # Middle layer - map prev_vertices to spatial positions
            # prev_vertices should be in order: [(h=0,w=0,c=0), (h=0,w=0,c=1), ..., (h=0,w=0,c=C_in-1), (h=0,w=1,c=0), ...]
            if len(prev_vertices) != H_in * W_in * C_in:
                raise ValueError(f"Expected {H_in * W_in * C_in} prev_vertices, got {len(prev_vertices)}")
            
            idx = 0
            for h in range(H_in):
                for w in range(W_in):
                    for c in range(C_in):
                        input_vertices[(h, w, c)] = prev_vertices[idx]
                        idx += 1
        
        # Step 2: Create parameter nodes (factor nodes)
        param_vertices = {}  # (k_h, k_w, c_in, c_out) -> vertex
        param_idx = 0
        for c_out in range(out_channels):
            for c_in in range(in_channels):
                for kh in range(k_h):
                    for kw in range(k_w):
                        v = g.add_vertex()
                        v_type[v] = "parameter"
                        v_layer[v] = 1
                        g.vp.neuron_idx[v] = param_idx
                        param_vertices[(kh, kw, c_in, c_out)] = v
                        
                        # Add self-loop with weight magnitude
                        weight_val = weights[c_out, c_in, kh, kw]
                        if abs(weight_val) >= self.weight_threshold:
                            e = g.add_edge(v, v)
                            e_weight[e] = abs(weight_val)
                        
                        param_idx += 1
        
        # Step 3: Create output activation vertices
        output_vertices = {}  # (h, w, c) -> vertex
        output_list = []  # For return value
        for h in range(H_out):
            for w in range(W_out):
                for c in range(out_channels):
                    v = g.add_vertex()
                    v_type[v] = "output"
                    v_layer[v] = 2
                    g.vp.neuron_idx[v] = c + out_channels * (w + W_out * h)
                    if self.include_spatial_info:
                        g.vp.spatial_h[v] = h
                        g.vp.spatial_w[v] = w
                    output_vertices[(h, w, c)] = v
                    output_list.append(v)
        
        # Step 4: Add structural edges (weight 1.0)
        # These connect inputs to parameters and parameters to outputs
        for h_out in range(H_out):
            for w_out in range(W_out):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                # Calculate input position
                                h_in = h_out * s_h - p_h + kh
                                w_in = w_out * s_w - p_w + kw
                                
                                # Check if within bounds
                                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                    # Get vertices
                                    input_v = input_vertices[(h_in, w_in, c_in)]
                                    param_v = param_vertices[(kh, kw, c_in, c_out)]
                                    output_v = output_vertices[(h_out, w_out, c_out)]
                                    
                                    # Check weight magnitude
                                    weight_val = weights[c_out, c_in, kh, kw]
                                    if abs(weight_val) >= self.weight_threshold:
                                        # Add structural edges
                                        # Input -> Parameter (always forward)
                                        e1 = g.add_edge(input_v, param_v)
                                        e_weight[e1] = 1.0
                                        
                                        # Parameter -> Output (direction depends on sign)
                                        if self.handle_negative_weights and weight_val < 0:
                                            e2 = g.add_edge(output_v, param_v)
                                        else:
                                            e2 = g.add_edge(param_v, output_v)
                                        e_weight[e2] = 1.0
        
        # Handle bias if present
        if layer.bias is not None:
            bias = layer.bias.detach().cpu().numpy()
            for c_out in range(out_channels):
                if abs(bias[c_out]) >= self.weight_threshold:
                    # Create bias parameter node
                    bias_v = g.add_vertex()
                    v_type[bias_v] = "bias_parameter"
                    v_layer[bias_v] = 1
                    g.vp.neuron_idx[bias_v] = -1 - c_out
                    
                    # Self-loop with bias magnitude
                    e = g.add_edge(bias_v, bias_v)
                    e_weight[e] = abs(bias[c_out])
                    
                    # Connect to all output positions for this channel
                    for h in range(H_out):
                        for w in range(W_out):
                            output_v = output_vertices[(h, w, c_out)]
                            if self.handle_negative_weights and bias[c_out] < 0:
                                e = g.add_edge(output_v, bias_v)
                            else:
                                e = g.add_edge(bias_v, output_v)
                            e_weight[e] = 1.0
        
        # Normalize weights if requested
        self._normalize_edge_weights(g, e_weight, self.normalization_type)
        
        return g, output_list
    
    def _build_conv1d_graph(self, layer: nn.Conv1d, 
                           input_shape: Tuple[int, int],
                           prev_vertices: Optional[List[int]] = None) -> Tuple[Graph, List[int]]:
        """Build factor graph for Conv1d layer."""
        # Conv1d is similar to Conv2d but with one spatial dimension
        # We can reuse Conv2d logic by treating it as Conv2d with height=1
        C_in, L_in = input_shape
        expanded_shape = (C_in, 1, L_in)  # Add dummy height dimension
        
        # Create a temporary Conv2d layer with same parameters
        temp_conv2d = nn.Conv2d(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=(1, layer.kernel_size[0]),
            stride=(1, layer.stride[0]),
            padding=(0, layer.padding[0])
        )
        
        # Copy weights (reshape from Conv1d to Conv2d format)
        with torch.no_grad():
            temp_conv2d.weight.data = layer.weight.data.unsqueeze(2)
            if layer.bias is not None:
                temp_conv2d.bias.data = layer.bias.data
        
        # Build graph using Conv2d logic
        g, output_vertices = self._build_conv2d_graph(temp_conv2d, expanded_shape, prev_vertices)
        
        # Remove the dummy spatial dimension from properties if needed
        if self.include_spatial_info and "spatial_h" in g.vp:
            # Set all h values to 0 (since it's 1D)
            for v in g.vertices():
                g.vp.spatial_h[v] = 0
        
        return g, output_vertices


class UnifiedGraphBuilder(NetworkGraphBuilder):
    """
    Unified graph builder that handles complete neural networks with mixed layer types.
    
    This class orchestrates the construction of a single graph representation
    for an entire neural network, handling transitions between different layer types.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize unified builder with sub-builders for each layer type."""
        super().__init__(*args, **kwargs)
        
        # Initialize specialized builders
        self.mlp_builder = MLPGraphBuilder(
            normalize_weights=False,  # We'll normalize at the end
            weight_threshold=self.weight_threshold,
            handle_negative_weights=self.handle_negative_weights,
            weight_encoding=self.weight_encoding,
            min_edge_distance=self.min_edge_distance,
            normalization_type=self.normalization_type
        )
        
        self.conv_builder = ConvGraphBuilder(
            normalize_weights=False,
            weight_threshold=self.weight_threshold,
            handle_negative_weights=self.handle_negative_weights,
            weight_encoding=self.weight_encoding,
            min_edge_distance=self.min_edge_distance,
            normalization_type=self.normalization_type,
            include_spatial_info=True
        )
    
    def _compute_max_abs_weight(self, model: nn.Module) -> float:
        """Compute maximum absolute weight across all layers."""
        max_abs_weight = 0.0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.detach().cpu().numpy()
                max_abs_weight = max(max_abs_weight, np.max(np.abs(weights)))
                if module.bias is not None:
                    bias = module.bias.detach().cpu().numpy()
                    max_abs_weight = max(max_abs_weight, np.max(np.abs(bias)))
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                weights = module.weight.detach().cpu().numpy()
                max_abs_weight = max(max_abs_weight, np.max(np.abs(weights)))
                if module.bias is not None:
                    bias = module.bias.detach().cpu().numpy()
                    max_abs_weight = max(max_abs_weight, np.max(np.abs(bias)))
        
        return max_abs_weight
    
    def build_network_graph(self, model: nn.Module) -> Graph:
        """
        Build complete graph representation of a neural network.
        
        Args:
            model: PyTorch model to convert to graph
            
        Returns:
            Graph representation of the network
        """
        # Compute max absolute weight for reverse encoding
        if self.weight_encoding == 'reverse':
            self.max_abs_weight = self._compute_max_abs_weight(model)
            self.mlp_builder.max_abs_weight = self.max_abs_weight
            self.conv_builder.max_abs_weight = self.max_abs_weight
        
        # Create main graph
        g, v_type, v_layer, e_weight = self._create_graph_with_properties()
        
        # Track current layer outputs and shape
        current_vertices = None
        current_shape = None
        layer_idx = 0
        
        # Get input shape from model if available
        if hasattr(model, 'input_shape'):
            current_shape = model.input_shape
        elif hasattr(model, 'config') and 'input_shape' in model.config:
            current_shape = model.config['input_shape']
        else:
            # Try to infer from first layer
            first_layer = next(model.modules())
            if isinstance(first_layer, nn.Linear):
                current_shape = (first_layer.in_features,)
            else:
                raise ValueError("Cannot infer input shape. Please provide it in the model.")
        
        # Process each layer
        for name, module in model.named_modules():
            # Skip the parent module and module lists
            if module is model or isinstance(module, nn.ModuleList):
                continue
                
            if isinstance(module, nn.Linear):
                # Handle Linear layers
                if current_vertices is not None and len(current_shape) > 1:
                    # Need to flatten first
                    current_vertices = self._flatten_vertices(g, current_vertices, 
                                                            current_shape, layer_idx)
                    current_shape = (np.prod(current_shape),)
                    layer_idx += 1
                
                # Use MLPGraphBuilder for proper encoding support
                mlp_g, mlp_outputs = self.mlp_builder.build_graph(
                    module, current_shape, current_vertices
                )
                
                # Merge into main graph if not the first layer
                if current_vertices is not None:
                    current_vertices = self._merge_subgraph(g, mlp_g, layer_idx)
                else:
                    # First layer - just use the mlp graph
                    g = mlp_g
                    current_vertices = mlp_outputs
                    v_type = g.vp.type
                    v_layer = g.vp.layer
                    e_weight = g.ep.weight
                current_shape = (module.out_features,)
                layer_idx += 1
                
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                # Handle Convolutional layers
                # For now, always create new graph and merge
                conv_g, conv_outputs = self.conv_builder.build_graph(
                    module, current_shape, None  # Don't pass prev_vertices to avoid issues
                )
                
                # Merge into main graph
                if current_vertices is not None:
                    # Need to connect previous layer outputs to conv inputs
                    current_vertices = self._merge_conv_subgraph(
                        g, conv_g, layer_idx, current_vertices, current_shape
                    )
                else:
                    # First layer - use the conv graph
                    g = conv_g
                    current_vertices = conv_outputs
                    v_type = g.vp.type
                    v_layer = g.vp.layer
                    e_weight = g.ep.weight
                
                # Update shape
                if isinstance(module, nn.Conv2d):
                    H_out = self._calculate_output_dim(current_shape[1], module.kernel_size[0], 
                                                     module.stride[0], module.padding[0])
                    W_out = self._calculate_output_dim(current_shape[2], module.kernel_size[1], 
                                                     module.stride[1], module.padding[1])
                    current_shape = (module.out_channels, H_out, W_out)
                else:  # Conv1d
                    L_out = self._calculate_output_dim(current_shape[1], module.kernel_size[0], 
                                                     module.stride[0], module.padding[0])
                    current_shape = (module.out_channels, L_out)
                
                layer_idx += 1
                
            elif isinstance(module, nn.Flatten):
                # Handle explicit flatten
                current_vertices = self._flatten_vertices(g, current_vertices, 
                                                        current_shape, layer_idx)
                current_shape = (np.prod(current_shape),)
                layer_idx += 1
                
            elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d)):
                # Handle pooling layers
                current_vertices = self._add_pooling_layer(g, module, current_vertices, 
                                                          current_shape, layer_idx)
                
                # Update shape
                if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                    H_out = self._calculate_output_dim(current_shape[1], module.kernel_size, 
                                                     module.stride, module.padding)
                    W_out = self._calculate_output_dim(current_shape[2], module.kernel_size, 
                                                     module.stride, module.padding)
                    current_shape = (current_shape[0], H_out, W_out)
                else:  # 1D pooling
                    L_out = self._calculate_output_dim(current_shape[1], module.kernel_size, 
                                                     module.stride, module.padding)
                    current_shape = (current_shape[0], L_out)
                
                layer_idx += 1
                
            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
                # Skip activation functions - they don't change graph structure
                pass
                
            elif hasattr(module, 'target_shape'):
                # Handle custom Reshape module
                if hasattr(module, 'target_shape'):
                    current_shape = tuple(module.target_shape)
                    # The vertices remain the same, just the interpretation changes
                
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # Handle batch normalization as identity edges with learned scaling
                current_vertices = self._add_batchnorm_layer(g, module, current_vertices, 
                                                            layer_idx)
                layer_idx += 1
        
        # Normalize all edge weights at the end
        if self.normalize_weights:
            self._normalize_edge_weights(g, e_weight, self.normalization_type)
        
        return g
    
    def _calculate_output_dim(self, input_dim: int, kernel_size: int, 
                            stride: int, padding: int) -> int:
        """Calculate output dimension for conv/pool operations."""
        return (input_dim + 2 * padding - kernel_size) // stride + 1
    
    def _merge_subgraph(self, main_graph: Graph, subgraph: Graph, 
                       layer_offset: int) -> List[int]:
        """Merge a subgraph into the main graph."""
        # Create vertex mapping
        vertex_map = {}
        output_vertices = []
        
        # Copy vertices
        for v in subgraph.vertices():
            new_v = main_graph.add_vertex()
            vertex_map[v] = new_v
            
            # Copy properties
            for prop_name, prop_map in subgraph.vp.items():
                if prop_name in main_graph.vp:
                    main_graph.vp[prop_name][new_v] = prop_map[v]
            
            # Adjust layer index
            main_graph.vp.layer[new_v] += layer_offset
            
            # Track output vertices
            if subgraph.vp.type[v] == "output":
                output_vertices.append(new_v)
        
        # Copy edges
        for e in subgraph.edges():
            src = vertex_map[e.source()]
            tgt = vertex_map[e.target()]
            new_e = main_graph.add_edge(src, tgt)
            
            # Copy edge weight
            if "weight" in subgraph.ep:
                main_graph.ep.weight[new_e] = subgraph.ep.weight[e]
        
        return output_vertices
    
    def _merge_conv_subgraph(self, main_graph: Graph, conv_graph: Graph,
                            layer_offset: int, prev_vertices: List[int],
                            input_shape: Tuple[int, ...]) -> List[int]:
        """Merge conv subgraph and connect to previous layer."""
        # Create vertex mapping
        vertex_map = {}
        output_vertices = []
        
        # Map conv graph input vertices to prev_vertices
        conv_input_vertices = []
        for v in conv_graph.vertices():
            if conv_graph.vp.type[v] == "input":
                conv_input_vertices.append(v)
        
        # Sort both by spatial position to ensure correct mapping
        conv_input_vertices.sort(key=lambda v: conv_graph.vp.neuron_idx[v])
        
        # Create mapping from conv input vertices to main graph prev vertices
        if len(conv_input_vertices) != len(prev_vertices):
            raise ValueError(f"Mismatch: {len(conv_input_vertices)} conv inputs vs {len(prev_vertices)} prev vertices")
        
        for conv_v, main_v in zip(conv_input_vertices, prev_vertices):
            vertex_map[conv_v] = main_v
        
        # Copy non-input vertices
        for v in conv_graph.vertices():
            if conv_graph.vp.type[v] != "input":
                new_v = main_graph.add_vertex()
                vertex_map[v] = new_v
                
                # Copy properties
                for prop_name, prop_map in conv_graph.vp.items():
                    if prop_name in main_graph.vp:
                        main_graph.vp[prop_name][new_v] = prop_map[v]
                
                # Adjust layer index
                main_graph.vp.layer[new_v] += layer_offset
                
                # Track output vertices
                if conv_graph.vp.type[v] == "output":
                    output_vertices.append(new_v)
        
        # Copy edges
        for e in conv_graph.edges():
            src = vertex_map[e.source()]
            tgt = vertex_map[e.target()]
            new_e = main_graph.add_edge(src, tgt)
            
            # Copy edge weight
            if "weight" in conv_graph.ep:
                main_graph.ep.weight[new_e] = conv_graph.ep.weight[e]
        
        return output_vertices
    
    def _add_linear_layer(self, g: Graph, layer: nn.Linear, 
                         prev_vertices: Optional[List[int]], 
                         input_shape: Tuple[int, ...], 
                         layer_idx: int) -> List[int]:
        """Add a linear layer directly to the main graph."""
        v_type = g.vp.type
        v_layer = g.vp.layer
        e_weight = g.ep.weight
        
        # Get layer parameters
        in_features = layer.in_features
        out_features = layer.out_features
        weights = layer.weight.detach().cpu().numpy()
        
        # Create or reuse input vertices
        if prev_vertices is None:
            # Create input vertices
            input_vertices = []
            for i in range(in_features):
                v = g.add_vertex()
                v_type[v] = "input"
                v_layer[v] = layer_idx
                g.vp.neuron_idx[v] = i
                input_vertices.append(v)
        else:
            input_vertices = prev_vertices
            
        # Create output vertices
        output_vertices = []
        for i in range(out_features):
            v = g.add_vertex()
            v_type[v] = "hidden" if layer_idx > 0 else "output"
            v_layer[v] = layer_idx + 1
            g.vp.neuron_idx[v] = i
            output_vertices.append(v)
        
        # Add edges with weights
        for i in range(out_features):
            for j in range(in_features):
                weight = weights[i, j]
                
                # Skip near-zero weights
                if abs(weight) < self.weight_threshold:
                    continue
                    
                self._add_edge_with_sign(g, input_vertices[j], output_vertices[i], 
                                       weight, e_weight)
        
        # Handle bias
        if layer.bias is not None:
            bias = layer.bias.detach().cpu().numpy()
            bias_vertex = g.add_vertex()
            v_type[bias_vertex] = "bias"
            v_layer[bias_vertex] = layer_idx
            g.vp.neuron_idx[bias_vertex] = -1
            
            for i in range(out_features):
                if abs(bias[i]) >= self.weight_threshold:
                    self._add_edge_with_sign(g, bias_vertex, output_vertices[i], 
                                           bias[i], e_weight)
        
        return output_vertices
    
    def _flatten_vertices(self, g: Graph, vertices: List[int], 
                         shape: Tuple[int, ...], layer_idx: int) -> List[int]:
        """Create identity edges to flatten spatial dimensions."""
        # For now, just return the same vertices
        # In a more complete implementation, we might want to reorder vertices
        return vertices
    
    def _add_pooling_layer(self, g: Graph, module: nn.Module, 
                          input_vertices: List[int], input_shape: Tuple[int, ...], 
                          layer_idx: int) -> List[int]:
        """Add pooling layer as edges with weight 1.0."""
        # Simplified: create identity edges
        # In practice, pooling would create edges from multiple inputs to one output
        return input_vertices
    
    def _add_batchnorm_layer(self, g: Graph, module: nn.Module, 
                           input_vertices: List[int], layer_idx: int) -> List[int]:
        """Add batch normalization as scaled identity edges."""
        # Get learned scale parameters
        if hasattr(module, 'weight') and module.weight is not None:
            scales = module.weight.detach().cpu().numpy()
        else:
            scales = np.ones(len(input_vertices))
        
        # Create output vertices with scaled edges
        output_vertices = []
        e_weight = g.ep.weight
        
        for i, (v_in, scale) in enumerate(zip(input_vertices, scales)):
            if abs(scale) < self.weight_threshold:
                # Skip near-zero scales
                continue
                
            v_out = g.add_vertex()
            g.vp.type[v_out] = "hidden"
            g.vp.layer[v_out] = layer_idx
            g.vp.neuron_idx[v_out] = i
            output_vertices.append(v_out)
            
            # Add scaled edge
            self._add_edge_with_sign(g, v_in, v_out, scale, e_weight)
        
        return output_vertices if output_vertices else input_vertices
    
    def build_graph(self, layer: nn.Module, input_shape: Tuple[int, ...], 
                   prev_vertices: Optional[List[int]] = None) -> Tuple[Graph, List[int]]:
        """Build graph for a single layer (delegates to specialized builders)."""
        if isinstance(layer, nn.Linear):
            return self.mlp_builder.build_graph(layer, input_shape, prev_vertices)
        elif isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            return self.conv_builder.build_graph(layer, input_shape, prev_vertices)
        else:
            raise NotImplementedError(f"Layer type {type(layer)} not supported")