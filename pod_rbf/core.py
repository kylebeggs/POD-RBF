"""
Core POD-RBF training and inference functions.

Pure functional interface for JAX autodiff compatibility.
"""

import jax.numpy as jnp
from jax import Array

from .decomposition import compute_pod_basis
from .rbf import build_collocation_matrix, build_inference_matrix
from .shape_optimization import find_optimal_shape_param
from .types import ModelState, TrainConfig, TrainResult


def _normalize_params(params: Array) -> Array:
    """Ensure params is 2D: (n_params, n_points)."""
    if params.ndim == 1:
        return params[None, :]
    return params


def train(
    snapshot: Array,
    train_params: Array,
    config: TrainConfig = TrainConfig(),
    shape_factor: float | None = None,
) -> TrainResult:
    """
    Train POD-RBF model.

    Parameters
    ----------
    snapshot : Array
        Solution snapshots, shape (n_samples, n_snapshots).
        Each column is a snapshot at a different parameter value.
    train_params : Array
        Parameter values, shape (n_snapshots,) or (n_params, n_snapshots).
    config : TrainConfig
        Training configuration.
    shape_factor : float, optional
        RBF shape parameter. If None, automatically optimized.

    Returns
    -------
    TrainResult
        Training result containing model state and diagnostics.
    """
    train_params = _normalize_params(jnp.asarray(train_params))
    snapshot = jnp.asarray(snapshot)
    n_params, n_snapshots = train_params.shape

    assert snapshot.shape[1] == n_snapshots, (
        f"Mismatch: {snapshot.shape[1]} snapshots vs {n_snapshots} params"
    )

    # Compute parameter ranges for normalization
    params_range = jnp.ptp(train_params, axis=1)

    # Determine decomposition method based on memory
    memory_gb = snapshot.nbytes / 1e9
    use_eig = memory_gb >= config.mem_limit_gb

    # Find optimal shape factor if not provided
    if shape_factor is None:
        shape_factor = find_optimal_shape_param(
            train_params,
            params_range,
            cond_range=config.cond_range,
            max_iters=config.max_bisection_iters,
            c_low_init=config.c_low_init,
            c_high_init=config.c_high_init,
            c_high_step=config.c_high_step,
            c_high_search_iters=config.c_high_search_iters,
        )

    # Compute truncated POD basis
    basis, cumul_energy, truncated_energy = compute_pod_basis(
        snapshot, config.energy_threshold, use_eig=use_eig
    )

    # Build collocation matrix and compute weights
    F = build_collocation_matrix(train_params, params_range, shape_factor)
    A = basis.T @ snapshot
    weights = A @ jnp.linalg.pinv(F.T)

    state = ModelState(
        basis=basis,
        weights=weights,
        shape_factor=float(shape_factor),
        train_params=train_params,
        params_range=params_range,
        truncated_energy=float(truncated_energy),
        cumul_energy=cumul_energy,
    )

    return TrainResult(
        state=state,
        n_modes=basis.shape[1],
        used_eig_decomp=use_eig,
    )


def inference(state: ModelState, inf_params: Array) -> Array:
    """
    Inference trained model at multiple parameter points.

    Parameters
    ----------
    state : ModelState
        Trained model state from train().
    inf_params : Array
        Inference parameters, shape (n_params, n_points) or (n_points,).

    Returns
    -------
    Array
        Predicted solutions, shape (n_samples, n_points).
    """
    inf_params = _normalize_params(jnp.asarray(inf_params))

    F = build_inference_matrix(
        state.train_params,
        inf_params,
        state.params_range,
        state.shape_factor,
    )

    A = state.weights @ F.T
    return state.basis @ A


def inference_single(state: ModelState, inf_param: Array) -> Array:
    """
    Inference trained model at a single parameter point.

    More convenient for gradient computation.

    Parameters
    ----------
    state : ModelState
        Trained model state from train().
    inf_param : Array
        Single inference parameter, scalar or shape (n_params,).

    Returns
    -------
    Array
        Predicted solution, shape (n_samples,).
    """
    inf_param = jnp.asarray(inf_param)

    # Handle scalar input
    if inf_param.ndim == 0:
        inf_param = inf_param[None]

    # Shape to (n_params, 1) for inference
    inf_params = inf_param[:, None]

    result = inference(state, inf_params)
    return result[:, 0]
