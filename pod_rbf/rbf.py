"""
Radial Basis Function (RBF) kernel and matrix construction.

Uses Hardy Inverse Multi-Quadrics (IMQ): phi(r) = 1 / sqrt(r^2/c^2 + 1)
"""

import jax
import jax.numpy as jnp
from jax import Array


def build_collocation_matrix(
    train_params: Array,
    params_range: Array,
    shape_factor: float,
) -> Array:
    """
    Build RBF collocation matrix for training.

    Parameters
    ----------
    train_params : Array
        Training parameters, shape (n_params, n_train_points).
    params_range : Array
        Range of each parameter for normalization, shape (n_params,).
    shape_factor : float
        RBF shape parameter c.

    Returns
    -------
    Array
        Collocation matrix, shape (n_train_points, n_train_points).
    """
    n_params, n_train = train_params.shape

    def accumulate_r2(i: int, r2: Array) -> Array:
        param_row = train_params[i, :]
        diff = param_row[:, None] - param_row[None, :]  # (n_train, n_train)
        return r2 + (diff / params_range[i]) ** 2

    r2 = jax.lax.fori_loop(0, n_params, accumulate_r2, jnp.zeros((n_train, n_train)))

    return 1.0 / jnp.sqrt(r2 / (shape_factor**2) + 1.0)


def build_inference_matrix(
    train_params: Array,
    inf_params: Array,
    params_range: Array,
    shape_factor: float,
) -> Array:
    """
    Build RBF inference matrix for prediction at new parameters.

    Parameters
    ----------
    train_params : Array
        Training parameters, shape (n_params, n_train_points).
    inf_params : Array
        Inference parameters, shape (n_params, n_inf_points).
    params_range : Array
        Range of each parameter for normalization, shape (n_params,).
    shape_factor : float
        RBF shape parameter c.

    Returns
    -------
    Array
        Inference matrix, shape (n_inf_points, n_train_points).
    """
    n_params = train_params.shape[0]
    n_inf = inf_params.shape[1]
    n_train = train_params.shape[1]

    def accumulate_r2(i: int, r2: Array) -> Array:
        diff = inf_params[i, :, None] - train_params[i, None, :]  # (n_inf, n_train)
        return r2 + (diff / params_range[i]) ** 2

    r2 = jax.lax.fori_loop(0, n_params, accumulate_r2, jnp.zeros((n_inf, n_train)))

    return 1.0 / jnp.sqrt(r2 / (shape_factor**2) + 1.0)
