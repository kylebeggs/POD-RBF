"""
POD-RBF: Proper Orthogonal Decomposition - Radial Basis Function Network.

A JAX-based implementation enabling autodifferentiation for:
- Gradient optimization
- Sensitivity analysis
- Inverse problems

Usage
-----
>>> import pod_rbf
>>> import jax.numpy as jnp
>>>
>>> # Train model
>>> result = pod_rbf.train(snapshot, params)
>>>
>>> # Inference
>>> pred = pod_rbf.inference_single(result.state, jnp.array(450.0))
>>>
>>> # Autodiff
>>> import jax
>>> grad_fn = jax.grad(lambda p: jnp.sum(pod_rbf.inference_single(result.state, p)**2))
>>> gradient = grad_fn(jnp.array(450.0))
"""

import jax

# Enable float64 for numerical stability (SVD, condition numbers)
jax.config.update("jax_enable_x64", True)

from .core import inference, inference_single, train
from .io import build_snapshot_matrix, load_model, save_model
from .types import ModelState, TrainConfig, TrainResult

try:
    from importlib.metadata import version

    __version__ = version("pod_rbf")
except Exception:
    __version__ = "unknown"

__all__ = [
    # Core functions
    "train",
    "inference",
    "inference_single",
    # Types
    "ModelState",
    "TrainConfig",
    "TrainResult",
    # I/O
    "build_snapshot_matrix",
    "save_model",
    "load_model",
]
