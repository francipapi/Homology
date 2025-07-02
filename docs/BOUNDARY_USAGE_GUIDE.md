# Decision Boundary Topology Analysis - Usage Guide

This guide explains how to use the two-step decision boundary topology analysis pipeline.

## 🎯 **Two-Step Process Overview**

### **Step 1: Train and Extract Boundaries**
- Uses `decision_boundary_trainer.py`
- Trains neural network
- Extracts decision boundary data (points/meshes) 
- **Does NOT compute topology** (for speed)
- Saves boundary data to files

### **Step 2: Compute Topology Separately**
- Uses `compute_boundary_homology.py`
- Loads saved boundary data
- Computes persistent homology using Ripser
- Saves topology results
- Much faster and can be parallelized

---

## 🚀 **Quick Start**

### **1. Train with Boundary Extraction**

```bash
# Train a single architecture with boundary extraction
python src/models/decision_boundary_trainer.py \
  --training-config configs/training_config.yaml \
  --boundary-config configs/decision_boundary_config.yaml
```

This will:
- Train your network for 50 epochs
- Extract decision boundaries every 5 epochs (configurable)
- Save boundary data to `results/decision_boundary_analysis/`
- **Skip topology computation** (much faster!)

### **2. Compute Topology from Saved Data**

```bash
# Compute topology for all extracted boundaries
python src/topology/compute_boundary_homology.py \
  --config configs/decision_boundary_config.yaml \
  --input-dir results/decision_boundary_analysis \
  --output-dir results/boundary_topology
```

This will:
- Load all saved boundary data
- Compute persistent homology using Ripser
- Save topology results and Betti numbers
- Create summary CSV files

---

## 📁 **File Structure After Running**

```
results/
├── decision_boundary_analysis/
│   ├── boundaries/              # Mesh files (.ply)
│   │   ├── boundary_epoch_0000.ply
│   │   ├── boundary_epoch_0005.ply
│   │   └── ...
│   ├── topology/               # Raw boundary points
│   │   ├── topology_epoch_0000.pt
│   │   ├── topology_epoch_0005.pt
│   │   └── ...
│   └── analysis/
│       └── complete_training_results.pt  # Everything together
└── boundary_topology/          # Topology computation results
    ├── boundary_topology_results.pt      # Complete results
    ├── boundary_topology_summary.csv     # Summary table
    └── *_boundary_topology.pt           # Per-architecture results
```

---

## ⚙️ **Configuration**

### **Key Settings in `decision_boundary_config.yaml`:**

```yaml
training:
  extraction_schedule:
    enabled: true
    frequency: 5          # Extract every 5 epochs
    start_epoch: 0        # Start immediately
    final_extraction: true

extraction:
  grid:
    resolution: [64, 64, 64]  # Grid resolution (affects speed)
  
  point_sampling:
    enabled: true
    num_points: 5000      # Number of boundary points to sample

topology:
  computation:
    enabled: false        # DISABLED during training!
    max_dimension: 2      # Compute H0, H1, H2
    max_edge_length: 1.0  # Topology threshold
```

---

## 🎨 **Visualization**

### **Create Visualizations:**

```bash
# Visualize decision boundaries
python src/visualization/decision_boundary_viz.py \
  results/decision_boundary_analysis/complete_training_results.pt \
  --animation --output boundary_evolution.html
```

### **Architecture Comparison:**

```bash
# Compare multiple architectures
python src/analysis/boundary_topology_comparison.py \
  --results arch1_results.pt arch2_results.pt \
  --labels "Architecture 1" "Architecture 2" \
  --output-dir comparison_results
```

---

## 🔧 **Troubleshooting**

### **Training Gets Stuck?**
- Set `topology.computation.enabled: false` in config
- This disables slow topology computation during training

### **Missing Boundary Data?**
- Check `results/decision_boundary_analysis/analysis/complete_training_results.pt`
- Look for `boundary_results` in the saved data

### **Topology Computation Fails?**
- Reduce `topology.sampling.num_points` to 1000 or fewer
- Set `topology.computation.max_dimension: 1` for speed

### **Out of Memory?**
- Reduce `extraction.grid.resolution` to `[32, 32, 32]`
- Reduce `extraction.point_sampling.num_points` to 2000

---

## 📊 **Example Workflow**

### **Compare Different Architectures:**

```bash
# Step 1: Train Architecture 1 (narrow & deep)
# Edit configs/training_config.yaml: num_hidden_layers=6, hidden_dim=16
python src/models/decision_boundary_trainer.py \
  --training-config configs/training_config.yaml \
  --boundary-config configs/decision_boundary_config.yaml \
  --save-model results/models/narrow_deep.pt

# Step 2: Train Architecture 2 (wide & shallow)  
# Edit configs/training_config.yaml: num_hidden_layers=2, hidden_dim=64
python src/models/decision_boundary_trainer.py \
  --training-config configs/training_config.yaml \
  --boundary-config configs/decision_boundary_config.yaml \
  --save-model results/models/wide_shallow.pt

# Step 3: Compute topology for both
python src/topology/compute_boundary_homology.py \
  --input-dir results/decision_boundary_analysis \
  --output-dir results/boundary_topology

# Step 4: Compare results
python src/analysis/boundary_topology_comparison.py \
  --results results/boundary_topology/*_boundary_topology.pt \
  --output-dir results/architecture_comparison \
  --report --plots
```

---

## 🎯 **Key Benefits of Two-Step Process**

1. **Training is Fast**: No topology computation during training
2. **Flexible**: Compute topology with different parameters later
3. **Parallel**: Topology computation can use multiple cores
4. **Reusable**: Same boundary data, different topology analyses
5. **Debuggable**: Easy to test each step independently

---

## 🐛 **Testing Individual Components**

```bash
# Test boundary extraction only
python -c "
from src.topology.decision_boundary_homology import test_boundary_extraction
test_boundary_extraction()
"

# Test topology computation only  
python debug_topology.py

# Test visualization only
python quick_viz_test.py
```

This two-step approach separates concerns and makes the pipeline much more efficient and flexible!