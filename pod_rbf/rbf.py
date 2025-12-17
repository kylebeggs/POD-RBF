"""
Radial Basis Function (RBF) kernel and matrix construction.

Supports multiple kernel types:
- Inverse Multi-Quadrics (IMQ): phi(r) = 1 / sqrt(r²/c² + 1)
- Gaussian: phi(r) = exp(-r²/c²)
- Polyharmonic Splines (PHS): phi(r) = r^k or r^k*log(r)
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import Array

from .kernels import apply_kernel


def build_collocation_matrix(
    train_params: Array,
    params_range: Array,
    kernel: str = "imq",
    shape_factor: float | None = None,
    kernel_order: int = 3,
) -> Array:
    """
    Build RBF collocation matrix for training.

    Parameters
    ----------
    train_params : Array
        Training parameters, shape (n_params, n_train_points).
    params_range : Array
        Range of each parameter for normalization, shape (n_params,).
    kernel : str, optional
        Kernel type: 'imq', 'gaussian', or 'polyharmonic_spline'.
        Default is 'imq'.
    shape_factor : float | None, optional
        RBF shape parameter c. Required for IMQ and Gaussian kernels.
        Ignored for polyharmonic splines.
    kernel_order : int, optional
        Order for polyharmonic splines (default 3).
        Ignored for other kernels.

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

    return apply_kernel(r2, kernel, shape_factor, kernel_order)


def build_inference_matrix(
    train_params: Array,
    inf_params: Array,
    params_range: Array,
    kernel: str = "imq",
    shape_factor: float | None = None,
    kernel_order: int = 3,
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
    kernel : str, optional
        Kernel type: 'imq', 'gaussian', or 'polyharmonic_spline'.
        Default is 'imq'.
    shape_factor : float | None, optional
        RBF shape parameter c. Required for IMQ and Gaussian kernels.
        Ignored for polyharmonic splines.
    kernel_order : int, optional
        Order for polyharmonic splines (default 3).
        Ignored for other kernels.

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

    return apply_kernel(r2, kernel, shape_factor, kernel_order)


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


def solve_augmented_system_direct(
    F: Array,
    P: Array,
    rhs: Array,
) -> tuple[Array, Array]:
    """
    Solve augmented RBF system by direct assembly and solve.

    Solves the saddle-point system:
        [F  P] [λ]   [rhs]
        [P.T 0] [c] = [0]

    This method assembles and solves the full system directly, which works
    for kernels where F is not positive definite (e.g., polyharmonic splines).

    Parameters
    ----------
    F : Array
        RBF collocation matrix, shape (n_train, n_train).
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
    n_train = F.shape[0]
    n_poly = P.shape[1]
    n_rhs = rhs.shape[0]

    # Assemble full augmented system matrix
    # [F  P ]
    # [P' 0 ]
    top = jnp.hstack([F, P])
    bottom = jnp.hstack([P.T, jnp.zeros((n_poly, n_poly))])
    A_aug = jnp.vstack([top, bottom])

    # Assemble augmented RHS
    # [rhs]
    # [0  ]
    rhs_aug = jnp.hstack([rhs, jnp.zeros((n_rhs, n_poly))])

    # Solve the full system
    solution = jnp.linalg.solve(A_aug, rhs_aug.T).T  # (n_rhs, n_train + n_poly)

    # Extract weights and polynomial coefficients
    rbf_weights = solution[:, :n_train]
    poly_coeffs = solution[:, n_train:]

    return rbf_weights, poly_coeffs
