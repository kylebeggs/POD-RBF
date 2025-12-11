"""Tests for POD basis decomposition."""

import jax.numpy as jnp
import numpy as np
import pytest

from pod_rbf.decomposition import (
    compute_pod_basis,
    compute_pod_basis_eig,
    compute_pod_basis_svd,
)


class TestComputePodBasisSvd:
    """Test SVD-based POD basis computation."""

    def test_orthonormality(self):
        """SVD-based basis should be orthonormal."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(100, 10))

        basis, _, _ = compute_pod_basis_svd(snapshot, 0.99)

        gram = basis.T @ basis
        identity = jnp.eye(gram.shape[0])
        assert jnp.allclose(gram, identity, atol=1e-10), "Basis should be orthonormal"

    def test_energy_threshold(self):
        """Retained energy should meet threshold."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(50, 10))

        basis, cumul_energy, truncated_energy = compute_pod_basis_svd(snapshot, 0.95)

        assert truncated_energy >= 0.95, f"Truncated energy {truncated_energy} < 0.95"

    def test_basis_shape(self):
        """Basis should have correct shape."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(100, 10))

        basis, _, _ = compute_pod_basis_svd(snapshot, 0.99)

        assert basis.shape[0] == 100, "Basis should have n_samples rows"
        assert basis.shape[1] <= 10, "Basis should have at most n_snapshots columns"
        assert basis.shape[1] >= 1, "Basis should have at least 1 column"

    def test_cumul_energy_monotonic(self):
        """Cumulative energy should be monotonically increasing."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(50, 15))

        _, cumul_energy, _ = compute_pod_basis_svd(snapshot, 0.99)

        diffs = jnp.diff(cumul_energy)
        assert jnp.all(diffs >= 0), "Cumulative energy should be monotonically increasing"

    def test_keep_all_energy(self):
        """With threshold >= 1, should keep all modes."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(50, 10))

        basis, _, truncated_energy = compute_pod_basis_svd(snapshot, 1.0)

        assert basis.shape[1] == 10, "Should keep all modes when threshold=1.0"
        assert jnp.isclose(truncated_energy, 1.0, atol=1e-10)


class TestComputePodBasisEig:
    """Test eigendecomposition-based POD basis computation."""

    def test_orthonormality(self):
        """Eig-based basis should be orthonormal."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(100, 10))

        basis, _, _ = compute_pod_basis_eig(snapshot, 0.99)

        gram = basis.T @ basis
        identity = jnp.eye(gram.shape[0])
        assert jnp.allclose(gram, identity, atol=1e-8), "Basis should be orthonormal"

    def test_energy_threshold(self):
        """Retained energy should meet threshold."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(50, 10))

        basis, cumul_energy, truncated_energy = compute_pod_basis_eig(snapshot, 0.95)

        assert truncated_energy >= 0.95, f"Truncated energy {truncated_energy} < 0.95"

    def test_basis_shape(self):
        """Basis should have correct shape."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(100, 10))

        basis, _, _ = compute_pod_basis_eig(snapshot, 0.99)

        assert basis.shape[0] == 100, "Basis should have n_samples rows"
        assert basis.shape[1] <= 10, "Basis should have at most n_snapshots columns"
        assert basis.shape[1] >= 1, "Basis should have at least 1 column"


class TestSvdEigEquivalence:
    """Test that SVD and eigendecomposition produce equivalent results."""

    def test_span_equivalence(self):
        """SVD and eig bases should span the same subspace."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(100, 10))

        basis_svd, _, _ = compute_pod_basis_svd(snapshot, 0.99)
        basis_eig, _, _ = compute_pod_basis_eig(snapshot, 0.99)

        # Same number of modes (may differ by 1 due to numerical differences)
        n_modes_svd = basis_svd.shape[1]
        n_modes_eig = basis_eig.shape[1]
        assert abs(n_modes_svd - n_modes_eig) <= 1, "Mode counts should be similar"

        # Project onto common subspace - projection matrices should be similar
        proj_svd = basis_svd @ basis_svd.T
        proj_eig = basis_eig @ basis_eig.T

        # They span similar subspaces if the projections are close
        # (accounting for potentially different number of modes)
        min_modes = min(n_modes_svd, n_modes_eig)
        basis_svd_trunc = basis_svd[:, :min_modes]
        basis_eig_trunc = basis_eig[:, :min_modes]
        proj_svd_trunc = basis_svd_trunc @ basis_svd_trunc.T
        proj_eig_trunc = basis_eig_trunc @ basis_eig_trunc.T

        assert jnp.allclose(proj_svd_trunc, proj_eig_trunc, atol=1e-6), "Projections should be similar"

    def test_energy_equivalence(self):
        """SVD and eig should both meet energy threshold."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(100, 10))

        _, _, energy_svd = compute_pod_basis_svd(snapshot, 0.95)
        _, _, energy_eig = compute_pod_basis_eig(snapshot, 0.95)

        # Both should meet threshold (may differ due to discrete truncation)
        assert energy_svd >= 0.95, f"SVD energy {energy_svd} below threshold"
        assert energy_eig >= 0.95, f"Eig energy {energy_eig} below threshold"


class TestComputePodBasis:
    """Test dispatch function."""

    def test_dispatch_svd(self):
        """use_eig=False should use SVD."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(50, 10))

        basis, _, _ = compute_pod_basis(snapshot, 0.99, use_eig=False)
        basis_svd, _, _ = compute_pod_basis_svd(snapshot, 0.99)

        assert jnp.allclose(basis, basis_svd), "use_eig=False should match SVD"

    def test_dispatch_eig(self):
        """use_eig=True should use eigendecomposition."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(50, 10))

        basis, _, _ = compute_pod_basis(snapshot, 0.99, use_eig=True)
        basis_eig, _, _ = compute_pod_basis_eig(snapshot, 0.99)

        assert jnp.allclose(basis, basis_eig), "use_eig=True should match eig"
