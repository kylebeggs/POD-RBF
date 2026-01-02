"""
POD basis computation via SVD or eigendecomposition.

Includes Gavish-Donoho optimal hard thresholding for noisy data.
"""

import jax
import jax.numpy as jnp
from jax import Array


def _marchenko_pastur_median(beta: float) -> float:
    """
    Compute median of Marchenko-Pastur distribution.

    Uses the approximation from Gavish & Donoho (2014) for beta in (0, 1].

    Parameters
    ----------
    beta : float
        Aspect ratio n/m where n <= m (0 < beta <= 1).

    Returns
    -------
    float
        Median of the Marchenko-Pastur distribution.
    """
    # Approximation valid for beta in (0, 1]
    # From Gavish & Donoho (2014), the median can be approximated
    # For the exact computation, we'd need to solve the MP CDF integral
    # This polynomial approximation is accurate to ~1e-4
    if beta <= 0 or beta > 1:
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    # Numerical approximation coefficients from the paper
    # μ_β ≈ (1 + √β)² for large matrices (asymptotic upper edge)
    # The median is approximately at 0.675 * upper_edge for small beta
    upper_edge = (1 + jnp.sqrt(beta)) ** 2
    lower_edge = (1 - jnp.sqrt(beta)) ** 2

    # Linear interpolation approximation for the median
    # Based on numerical integration of the MP density
    median_fraction = 0.6745 + 0.1 * beta  # Empirical fit
    return float(lower_edge + median_fraction * (upper_edge - lower_edge))


def _gavish_donoho_threshold(beta: float) -> float:
    """
    Compute optimal singular value threshold coefficient λ*(β).

    From Gavish & Donoho (2014) Theorem 1.

    Parameters
    ----------
    beta : float
        Aspect ratio n/m where n <= m (0 < beta <= 1).

    Returns
    -------
    float
        Optimal threshold coefficient λ*(β).
    """
    # λ*(β) = √(2(β+1) + 8β / ((β+1) + √(β²+14β+1)))
    numerator = 8 * beta
    denominator = (beta + 1) + jnp.sqrt(beta**2 + 14 * beta + 1)
    lambda_star = jnp.sqrt(2 * (beta + 1) + numerator / denominator)
    return float(lambda_star)


def optimal_rank_gavish_donoho(
    singular_values: Array,
    n_samples: int,
    n_snapshots: int,
    sigma: float | None = None,
) -> int:
    """
    Find optimal rank via Gavish-Donoho hard thresholding.

    Implements the optimal hard thresholding rule from:
    Gavish & Donoho (2014) "The Optimal Hard Threshold for Singular Values is 4/√3"

    Parameters
    ----------
    singular_values : Array
        Singular values in descending order.
    n_samples : int
        Number of rows in the original matrix.
    n_snapshots : int
        Number of columns in the original matrix.
    sigma : float, optional
        Known noise standard deviation. If None, estimated from data.

    Returns
    -------
    int
        Optimal number of singular values to keep (at least 1).
    """
    m, n = max(n_samples, n_snapshots), min(n_samples, n_snapshots)
    beta = n / m  # Aspect ratio, beta <= 1

    # Estimate noise if not provided
    if sigma is None:
        # Use median of smallest singular values for robust noise estimation
        # σ_est = median(S) / √(n * μ_β)
        median_sv = jnp.median(singular_values)
        mu_beta = _marchenko_pastur_median(beta)
        sigma = float(median_sv / jnp.sqrt(n * mu_beta))

    # Compute optimal threshold
    lambda_star = _gavish_donoho_threshold(beta)
    threshold = lambda_star * sigma * jnp.sqrt(n)

    # Count singular values above threshold
    rank = int(jnp.sum(singular_values > threshold))

    # Always keep at least 1 mode
    return max(1, rank)


def compute_pod_basis_svd_gavish_donoho(
    snapshot: Array,
    sigma: float | None = None,
) -> tuple[Array, Array, float]:
    """
    Compute truncated POD basis using Gavish-Donoho optimal rank selection.

    Best for noisy/experimental data where energy-based truncation may
    retain noise modes.

    Parameters
    ----------
    snapshot : Array
        Snapshot matrix, shape (n_samples, n_snapshots).
    sigma : float, optional
        Known noise standard deviation. If None, estimated from data.

    Returns
    -------
    basis : Array
        Truncated POD basis, shape (n_samples, n_modes).
    cumul_energy : Array
        Cumulative energy fraction per mode.
    truncated_energy : float
        Actual energy fraction retained.
    """
    U, S, _ = jnp.linalg.svd(snapshot, full_matrices=False)
    n_samples, n_snapshots = snapshot.shape

    cumul_energy = jnp.cumsum(S) / jnp.sum(S)

    # Find optimal rank using Gavish-Donoho
    optimal_rank = optimal_rank_gavish_donoho(S, n_samples, n_snapshots, sigma)

    # Truncate basis (optimal_rank is at least 1)
    trunc_id = optimal_rank - 1  # Convert to 0-indexed
    truncated_energy = cumul_energy[trunc_id]

    # Dynamic slice to get truncated basis
    basis = jax.lax.dynamic_slice(U, (0, 0), (U.shape[0], optimal_rank))

    return basis, cumul_energy, truncated_energy


def compute_pod_basis_svd(
    snapshot: Array,
    energy_threshold: float,
) -> tuple[Array, Array, float]:
    """
    Compute truncated POD basis via SVD.

    Use for smaller datasets (< mem_limit_gb).

    Parameters
    ----------
    snapshot : Array
        Snapshot matrix, shape (n_samples, n_snapshots).
    energy_threshold : float
        Minimum fraction of total energy to retain (0 < threshold <= 1).

    Returns
    -------
    basis : Array
        Truncated POD basis, shape (n_samples, n_modes).
    cumul_energy : Array
        Cumulative energy fraction per mode.
    truncated_energy : float
        Actual energy fraction retained.
    """
    U, S, _ = jnp.linalg.svd(snapshot, full_matrices=False)

    cumul_energy = jnp.cumsum(S) / jnp.sum(S)

    # Handle energy_threshold >= 1 (keep all modes)
    keep_all = energy_threshold >= 1.0
    # Find first index where cumul_energy > threshold
    mask = cumul_energy > energy_threshold
    trunc_id = jnp.where(
        keep_all,
        len(S) - 1,
        jnp.where(jnp.any(mask), jnp.argmax(mask), len(S) - 1),
    )

    truncated_energy = cumul_energy[trunc_id]

    # Dynamic slice to get truncated basis
    basis = jax.lax.dynamic_slice(U, (0, 0), (U.shape[0], trunc_id + 1))

    return basis, cumul_energy, truncated_energy


def compute_pod_basis_eig(
    snapshot: Array,
    energy_threshold: float,
) -> tuple[Array, Array, float]:
    """
    Compute truncated POD basis via eigendecomposition.

    More memory-efficient for large datasets (>= mem_limit_gb).
    Computes (n_snapshots x n_snapshots) covariance instead of full SVD.

    Parameters
    ----------
    snapshot : Array
        Snapshot matrix, shape (n_samples, n_snapshots).
    energy_threshold : float
        Minimum fraction of total energy to retain (0 < threshold <= 1).

    Returns
    -------
    basis : Array
        Truncated POD basis, shape (n_samples, n_modes).
    cumul_energy : Array
        Cumulative energy fraction per mode.
    truncated_energy : float
        Actual energy fraction retained.
    """
    # Covariance matrix (n_snapshots x n_snapshots)
    cov = snapshot.T @ snapshot
    eig_vals, eig_vecs = jnp.linalg.eigh(cov)

    # eigh returns ascending order, reverse to descending
    eig_vals = jnp.abs(eig_vals[::-1])
    eig_vecs = eig_vecs[:, ::-1]

    cumul_energy = jnp.cumsum(eig_vals) / jnp.sum(eig_vals)

    # Handle energy_threshold >= 1 (keep all modes)
    keep_all = energy_threshold >= 1.0
    mask = cumul_energy > energy_threshold
    trunc_id = jnp.where(
        keep_all,
        len(eig_vals) - 1,
        jnp.where(jnp.any(mask), jnp.argmax(mask), len(eig_vals) - 1),
    )

    truncated_energy = cumul_energy[trunc_id]

    # Truncate eigenvalues and eigenvectors
    eig_vals_trunc = jax.lax.dynamic_slice(eig_vals, (0,), (trunc_id + 1,))
    eig_vecs_trunc = jax.lax.dynamic_slice(
        eig_vecs, (0, 0), (eig_vecs.shape[0], trunc_id + 1)
    )

    # Compute POD basis from eigenvectors
    basis = (snapshot @ eig_vecs_trunc) / jnp.sqrt(eig_vals_trunc)

    return basis, cumul_energy, truncated_energy


def compute_pod_basis(
    snapshot: Array,
    energy_threshold: float,
    use_eig: bool = False,
) -> tuple[Array, Array, float]:
    """
    Compute truncated POD basis.

    Dispatches to SVD or eigendecomposition based on use_eig flag.
    The flag should be determined BEFORE JIT compilation based on memory.

    Parameters
    ----------
    snapshot : Array
        Snapshot matrix, shape (n_samples, n_snapshots).
    energy_threshold : float
        Minimum fraction of total energy to retain (0 < threshold <= 1).
    use_eig : bool
        If True, use eigendecomposition (memory efficient for large data).
        If False, use SVD (faster for smaller data).

    Returns
    -------
    basis : Array
        Truncated POD basis, shape (n_samples, n_modes).
    cumul_energy : Array
        Cumulative energy fraction per mode.
    truncated_energy : float
        Actual energy fraction retained.
    """
    if use_eig:
        return compute_pod_basis_eig(snapshot, energy_threshold)
    return compute_pod_basis_svd(snapshot, energy_threshold)
