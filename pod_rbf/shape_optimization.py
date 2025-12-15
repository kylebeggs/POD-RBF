"""
Shape parameter optimization for RBF interpolation.

Uses fixed-iteration bisection to find optimal shape parameter c such that
the collocation matrix condition number falls within a target range.
"""

import jax
import jax.numpy as jnp
from jax import Array

from .kernels import KERNEL_SHAPE_DEFAULTS
from .rbf import build_collocation_matrix


def find_optimal_shape_param(
    train_params: Array,
    params_range: Array,
    kernel: str = "imq",
    kernel_order: int = 3,
    cond_range: tuple[float, float] = (1e11, 1e12),
    max_iters: int = 50,
    c_low_init: float | None = None,
    c_high_init: float | None = None,
    c_high_step: float | None = None,
    c_high_search_iters: int = 200,
) -> float | None:
    """
    Find optimal RBF shape parameter via fixed-iteration bisection.

    Target: condition number in [cond_range[0], cond_range[1]].

    Parameters
    ----------
    train_params : Array
        Training parameters, shape (n_params, n_train_points).
    params_range : Array
        Range of each parameter for normalization, shape (n_params,).
    kernel : str, optional
        Kernel type: 'imq', 'gaussian', or 'polyharmonic_spline'.
        Default is 'imq'.
    kernel_order : int, optional
        Order for polyharmonic splines (default 3).
        Ignored for other kernels.
    cond_range : tuple
        Target condition number range (lower, upper).
    max_iters : int
        Maximum bisection iterations.
    c_low_init : float | None, optional
        Initial lower bound for shape parameter.
        If None, uses kernel-specific default.
    c_high_init : float | None, optional
        Initial upper bound for shape parameter.
        If None, uses kernel-specific default.
    c_high_step : float | None, optional
        Step size for expanding upper bound search.
        If None, uses kernel-specific default.
    c_high_search_iters : int
        Maximum iterations for upper bound search.

    Returns
    -------
    float | None
        Optimal shape parameter.
        Returns None for kernels that don't use shape parameters (e.g., PHS).
    """
    # PHS doesn't use shape parameters
    if kernel == "polyharmonic_spline":
        return None

    # Use kernel-specific defaults if not provided
    defaults = KERNEL_SHAPE_DEFAULTS.get(kernel, KERNEL_SHAPE_DEFAULTS["imq"])
    c_low_init = c_low_init or defaults["c_low"]
    c_high_init = c_high_init or defaults["c_high"]
    c_high_step = c_high_step or defaults["c_step"]

    cond_low, cond_high = cond_range

    # Step 1: Find upper bound where cond >= cond_low
    def search_c_high_iter(i: int, carry: tuple) -> tuple:
        c_high, found = carry
        C = build_collocation_matrix(
            train_params, params_range, kernel, c_high, kernel_order
        )
        cond = jnp.linalg.cond(C)
        should_continue = (~found) & (cond < cond_low)
        new_c_high = jnp.where(should_continue, c_high + c_high_step, c_high)
        new_found = found | (cond >= cond_low)
        return (new_c_high, new_found)

    c_high, _ = jax.lax.fori_loop(
        0, c_high_search_iters, search_c_high_iter, (c_high_init, False)
    )

    # Step 2: Bisection to find optimal c in range
    def bisection_iter(i: int, carry: tuple) -> tuple:
        c_low_bound, c_high_bound, optim_c, found = carry

        mid_c = (c_low_bound + c_high_bound) / 2.0
        C = build_collocation_matrix(
            train_params, params_range, kernel, mid_c, kernel_order
        )
        cond = jnp.linalg.cond(C)

        # Check if condition number is in target range
        in_range = (cond >= cond_low) & (cond <= cond_high)
        below_range = cond < cond_low

        # Update bounds based on condition number (only if not yet found)
        new_c_low = jnp.where(below_range & ~found, mid_c, c_low_bound)
        new_c_high = jnp.where((~below_range) & (~in_range) & ~found, mid_c, c_high_bound)
        new_optim_c = jnp.where(in_range & ~found, mid_c, optim_c)
        new_found = found | in_range

        return (new_c_low, new_c_high, new_optim_c, new_found)

    initial_guess = (c_low_init + c_high) / 2.0
    _, _, optim_c, _ = jax.lax.fori_loop(
        0, max_iters, bisection_iter, (c_low_init, c_high, initial_guess, False)
    )

    return optim_c
