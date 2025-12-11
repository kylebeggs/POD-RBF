# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

POD-RBF is a JAX-based Python library for building Reduced Order Models (ROMs) using Proper Orthogonal Decomposition combined with Radial Basis Function interpolation. It enables autodifferentiation for gradient optimization, sensitivity analysis, and inverse problems.

## Development Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run single test file
pytest tests/test_core.py -v

# Run specific test
pytest tests/test_core.py::TestGradients::test_inverse_problem -v
```

## Architecture

### Module Structure

```
pod_rbf/
    __init__.py           # Public API, enables float64
    types.py              # ModelState, TrainConfig, TrainResult (NamedTuples)
    core.py               # train(), inference(), inference_single()
    rbf.py                # build_collocation_matrix(), build_inference_matrix()
    decomposition.py      # compute_pod_basis_svd(), compute_pod_basis_eig()
    shape_optimization.py # find_optimal_shape_param() (fixed-iteration bisection)
    io.py                 # build_snapshot_matrix(), save_model(), load_model()
```

### Key Types

```python
class ModelState(NamedTuple):
    basis: Array              # (n_samples, n_modes)
    weights: Array            # (n_modes, n_train_points)
    shape_factor: float
    train_params: Array       # (n_params, n_train_points)
    params_range: Array       # (n_params,)
    truncated_energy: float
    cumul_energy: Array

class TrainConfig(NamedTuple):
    energy_threshold: float = 0.99
    mem_limit_gb: float = 16.0
    cond_range: tuple = (1e11, 1e12)
    max_bisection_iters: int = 50
```

### API

```python
import pod_rbf
import jax
import jax.numpy as jnp

# Train with default config (energy_threshold=0.99)
result = pod_rbf.train(snapshot, params)
state = result.state

# Train with custom config
config = pod_rbf.TrainConfig(energy_threshold=0.9)
result = pod_rbf.train(snapshot, params, config)

# Inference (single point)
pred = pod_rbf.inference_single(state, jnp.array(450.0))

# Inference (batch)
preds = pod_rbf.inference(state, jnp.array([400.0, 450.0, 500.0]))

# Autodiff
grad_fn = jax.grad(lambda p: jnp.sum(pod_rbf.inference_single(state, p)**2))
gradient = grad_fn(jnp.array(450.0))

# I/O
snapshot = pod_rbf.build_snapshot_matrix("data/train/")  # load CSVs from directory
pod_rbf.save_model("model.pkl", state)
state = pod_rbf.load_model("model.pkl")
```

### Data Shape Conventions

- **Snapshot matrix**: `(n_samples, n_snapshots)` - each column is one parameter's solution
- **Parameters**: 1D `(n_snapshots,)` or 2D `(n_params, n_snapshots)`
- **Inference output**: `(n_samples,)` for single, `(n_samples, n_points)` for batch

### Key Algorithms

1. **POD truncation**: Keeps modes until cumulative energy exceeds `energy_threshold`
2. **RBF kernel**: Hardy Inverse Multi-Quadrics: `1/√(r²/c² + 1)`
3. **Shape optimization**: Fixed-iteration bisection (50 iters) for condition number in [10^11, 10^12]

## Dependencies

- `jax>=0.4.0`, `jaxlib>=0.4.0` - Autodiff and JIT compilation
- `numpy` - File I/O operations
- `tqdm` - Progress bars
- Python ≥ 3.10
