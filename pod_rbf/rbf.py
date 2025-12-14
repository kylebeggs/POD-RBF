"""
Radial Basis Function (RBF) kernel and matrix construction.

Uses Hardy Inverse Multi-Quadrics (IMQ): phi(r) = 1 / sqrt(r^2/c^2 + 1)
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
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


def build_polynomial_basis(
    params: Array,
    params_range: Array,
    degree: int = 2,
) -> Array:
    """
    Build polynomial basis matrix for RBF augmentation.

    Parameters
    ----------
    params : Array
        Parameters, shape (n_params, n_points).
    params_range : Array
        Range of each parameter for normalization, shape (n_params,).
    degree : int
        Polynomial degree (0=constant, 1=linear, 2=quadratic).
        Note: This should be a Python int, not a traced value.

    Returns
    -------
    Array
        Polynomial basis matrix, shape (n_points, n_poly).
        - degree 0: [1] -> 1 column
        - degree 1: [1, p1, p2, ...] -> n_params + 1 columns
        - degree 2: [1, p1, ..., pn, p1², ..., pn², p1*p2, ...] -> (n+1)(n+2)/2 cols
    """
    n_params, n_points = params.shape

    # Normalize parameters by dividing by range (consistent between train/inference)
    p_norm = params / params_range[:, None]

    # Build polynomial terms - degree must be a Python int for JIT compatibility
    terms = [jnp.ones((n_points,))]  # constant term

    if degree >= 1:
        # Linear terms
        for i in range(n_params):
            terms.append(p_norm[i, :])

    if degree >= 2:
        # Squared terms
        for i in range(n_params):
            terms.append(p_norm[i, :] ** 2)
        # Cross terms
        for i in range(n_params):
            for j in range(i + 1, n_params):
                terms.append(p_norm[i, :] * p_norm[j, :])

    return jnp.stack(terms, axis=1)


def solve_augmented_system_schur(
    F: Array,
    P: Array,
    rhs: Array,
) -> tuple[Array, Array]:
    """
    Solve augmented RBF system via Schur complement.

    Solves the saddle-point system:
        [F  P] [λ]   [rhs]
        [P.T 0] [c] = [0]

    Using Schur complement: S = P.T @ F^{-1} @ P

    Parameters
    ----------
    F : Array
        RBF collocation matrix, shape (n_train, n_train). Symmetric positive definite.
    P : Array
        Polynomial basis matrix, shape (n_train, n_poly).
    rhs : Array
        Right-hand side, shape (n_rhs, n_train). Each row is a separate RHS.

    Returns
    -------
    tuple[Array, Array]
        rbf_weights : shape (n_rhs, n_train)
        poly_coeffs : shape (n_rhs, n_poly)
    """
    # Cholesky factorization of F (symmetric positive definite)
    cho_F = jla.cho_factor(F)

    # Solve F @ X = P for X = F^{-1} @ P
    F_inv_P = jla.cho_solve(cho_F, P)  # (n_train, n_poly)

    # Schur complement: S = P.T @ F^{-1} @ P
    S = P.T @ F_inv_P  # (n_poly, n_poly)

    # Solve F @ Y = rhs.T for Y = F^{-1} @ rhs.T
    F_inv_rhs = jla.cho_solve(cho_F, rhs.T)  # (n_train, n_rhs)

    # Solve S @ c.T = P.T @ F^{-1} @ rhs.T for polynomial coefficients
    schur_rhs = P.T @ F_inv_rhs  # (n_poly, n_rhs)
    poly_coeffs = jnp.linalg.solve(S, schur_rhs).T  # (n_rhs, n_poly)

    # Back-substitute: λ = F^{-1} @ (rhs - P @ c)
    rbf_weights = (F_inv_rhs - F_inv_P @ poly_coeffs.T).T  # (n_rhs, n_train)

    return rbf_weights, poly_coeffs
