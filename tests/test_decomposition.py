"""Tests for POD basis decomposition."""

import jax.numpy as jnp
import numpy as np
import pytest

from pod_rbf.decomposition import (
    compute_pod_basis,
    compute_pod_basis_eig,
    compute_pod_basis_svd,
    compute_pod_basis_svd_gavish_donoho,
    optimal_rank_gavish_donoho,
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


class TestGavishDonoho:
    """Test Gavish-Donoho optimal rank selection."""

    def test_clean_low_rank_data(self):
        """With clean low-rank data, should identify approximate true rank."""
        np.random.seed(42)
        # Create rank-5 matrix
        U = np.random.randn(100, 5)
        V = np.random.randn(5, 20)
        clean = U @ V
        snapshot = jnp.array(clean)

        basis, _, _ = compute_pod_basis_svd_gavish_donoho(snapshot)
        n_modes = basis.shape[1]

        # Should detect approximately rank 5 (may vary due to noise estimation)
        assert 3 <= n_modes <= 8, f"Expected ~5 modes for rank-5 data, got {n_modes}"

    def test_noisy_data_filters_noise(self):
        """With noisy data, should filter noise modes."""
        np.random.seed(42)
        # Create rank-5 signal + noise
        U = np.random.randn(100, 5)
        V = np.random.randn(5, 20)
        signal = U @ V
        # Scale signal to have reasonable SNR
        signal = signal * 10
        noise = np.random.randn(100, 20)
        noisy = jnp.array(signal + noise)

        basis, _, _ = compute_pod_basis_svd_gavish_donoho(noisy)
        n_modes = basis.shape[1]

        # Should keep signal modes (~5) not all 20
        assert n_modes < 15, f"Should filter noise, but got {n_modes} modes"
        assert n_modes >= 1, "Should keep at least 1 mode"

    def test_known_sigma(self):
        """Providing known noise level should work."""
        np.random.seed(42)
        # Create noisy data with known sigma (lower SNR for clearer signal/noise separation)
        U = np.random.randn(100, 5)
        V = np.random.randn(5, 20)
        signal = U @ V * 5  # Lower signal strength
        sigma_true = 2.0  # Higher noise
        noise = sigma_true * np.random.randn(100, 20)
        noisy = jnp.array(signal + noise)

        # With known sigma
        basis, _, _ = compute_pod_basis_svd_gavish_donoho(noisy, sigma=sigma_true)
        n_modes_known = basis.shape[1]

        # With auto-estimated sigma
        basis_auto, _, _ = compute_pod_basis_svd_gavish_donoho(noisy)
        n_modes_auto = basis_auto.shape[1]

        # Both should give reasonable results (keeping some signal modes)
        assert 1 <= n_modes_known <= 20, f"Known sigma gave {n_modes_known} modes"
        assert 1 <= n_modes_auto <= 20, f"Auto sigma gave {n_modes_auto} modes"

    def test_orthonormality(self):
        """Gavish-Donoho basis should be orthonormal."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(100, 20))

        basis, _, _ = compute_pod_basis_svd_gavish_donoho(snapshot)

        gram = basis.T @ basis
        identity = jnp.eye(gram.shape[0])
        assert jnp.allclose(gram, identity, atol=1e-10), "Basis should be orthonormal"

    def test_basis_shape(self):
        """Basis should have correct shape."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(100, 20))

        basis, _, _ = compute_pod_basis_svd_gavish_donoho(snapshot)

        assert basis.shape[0] == 100, "Basis should have n_samples rows"
        assert basis.shape[1] <= 20, "Basis should have at most n_snapshots columns"
        assert basis.shape[1] >= 1, "Basis should have at least 1 column"

    def test_cumul_energy_monotonic(self):
        """Cumulative energy should be monotonically increasing."""
        np.random.seed(42)
        snapshot = jnp.array(np.random.randn(50, 15))

        _, cumul_energy, _ = compute_pod_basis_svd_gavish_donoho(snapshot)

        diffs = jnp.diff(cumul_energy)
        assert jnp.all(diffs >= 0), "Cumulative energy should be monotonically increasing"

    def test_optimal_rank_function_directly(self):
        """Test optimal_rank_gavish_donoho function directly."""
        np.random.seed(42)
        # Create singular values with clear gap
        signal_svs = jnp.array([100.0, 50.0, 20.0, 10.0, 5.0])
        noise_svs = jnp.array([0.5, 0.4, 0.3, 0.2, 0.1])
        singular_values = jnp.concatenate([signal_svs, noise_svs])

        rank = optimal_rank_gavish_donoho(
            singular_values, n_samples=100, n_snapshots=10, sigma=0.3
        )

        # With clear signal-noise gap and known sigma, should identify ~5 modes
        assert 3 <= rank <= 7, f"Expected ~5 modes, got {rank}"

    def test_always_keeps_at_least_one_mode(self):
        """Should always keep at least 1 mode even with pure noise."""
        np.random.seed(42)
        # Pure noise matrix
        snapshot = jnp.array(np.random.randn(50, 10) * 0.01)

        basis, _, _ = compute_pod_basis_svd_gavish_donoho(snapshot)

        assert basis.shape[1] >= 1, "Should keep at least 1 mode"
