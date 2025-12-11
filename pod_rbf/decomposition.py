"""
POD basis computation via SVD or eigendecomposition.
"""

import jax
import jax.numpy as jnp
from jax import Array


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
