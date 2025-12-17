# Getting Started

## Installation

Install POD-RBF using pip:

```bash
pip install pod-rbf
```

Or using uv:

```bash
uv add pod-rbf
```

## Basic Workflow

POD-RBF follows a simple three-step workflow:

1. **Build a snapshot matrix** - Collect solution data at different parameter values
2. **Train the model** - Compute POD basis and RBF interpolation weights
3. **Inference** - Predict solutions at new parameter values

## Minimal Example

```python
import pod_rbf
import jax.numpy as jnp
import numpy as np

# 1. Define training parameters
params = np.array([1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

# 2. Build snapshot matrix from CSV files
#    Each CSV file contains one solution snapshot
snapshot = pod_rbf.build_snapshot_matrix("path/to/data/")

# 3. Train the model
result = pod_rbf.train(snapshot, params)

# 4. Predict at a new parameter value
prediction = pod_rbf.inference_single(result.state, jnp.array(450.0))
```

## Understanding the Snapshot Matrix

The snapshot matrix `X` has shape `(n_samples, n_snapshots)`:

- Each **column** is one solution snapshot at a specific parameter value
- Each **row** corresponds to a spatial location or degree of freedom
- `n_samples` is the number of points in your solution (e.g., mesh nodes)
- `n_snapshots` is the number of parameter values you trained on

For example, if you solve a problem on a 400-node mesh at 10 different parameter values, your snapshot matrix is `(400, 10)`.

!!! note "Parameter Ordering"
    The order of columns in the snapshot matrix must match the order of your parameter array. If column 5 contains the solution at Re=500, then `params[4]` must equal 500.

## What's Next?

- [Training Models](user-guide/training.md) - Learn about training configuration options
- [Inference](user-guide/inference.md) - Single-point and batch predictions
- [Autodifferentiation](user-guide/autodiff.md) - Use JAX for gradients and optimization
