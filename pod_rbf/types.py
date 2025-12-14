"""
Data structures for POD-RBF.

All types are NamedTuples for JAX pytree compatibility.
"""

from typing import NamedTuple

from jax import Array


class TrainConfig(NamedTuple):
    """Immutable training configuration."""

    energy_threshold: float = 0.99
    mem_limit_gb: float = 16.0
    cond_range: tuple[float, float] = (1e11, 1e12)
    max_bisection_iters: int = 50
    c_low_init: float = 0.011
    c_high_init: float = 1.0
    c_high_step: float = 0.01
    c_high_search_iters: int = 200
    poly_degree: int = 2  # Polynomial augmentation degree (0=none, 1=linear, 2=quadratic)


class ModelState(NamedTuple):
    """Immutable trained model state - a valid JAX pytree."""

    basis: Array  # Truncated POD basis (n_samples, n_modes)
    weights: Array  # RBF network weights (n_modes, n_train_points)
    shape_factor: float  # Optimized RBF shape parameter
    train_params: Array  # Training parameters (n_params, n_train_points)
    params_range: Array  # Parameter ranges for normalization (n_params,)
    truncated_energy: float  # Energy retained after truncation
    cumul_energy: Array  # Cumulative energy per mode
    poly_coeffs: Array | None  # Polynomial coefficients (n_modes, n_poly) or None
    poly_degree: int  # Polynomial degree used (0=none)


class TrainResult(NamedTuple):
    """Result from training, includes diagnostics."""

    state: ModelState
    n_modes: int
    used_eig_decomp: bool  # True if eigendecomposition was used
