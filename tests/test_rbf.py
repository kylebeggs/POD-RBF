"""Tests for RBF matrix construction."""

import jax.numpy as jnp
import pytest

from pod_rbf.rbf import build_collocation_matrix, build_inference_matrix


class TestBuildCollocationMatrix:
    """Test RBF collocation matrix construction."""

    def test_symmetry(self):
        """Collocation matrix should be symmetric."""
        params = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        params_range = jnp.array([4.0])
        C = build_collocation_matrix(params, params_range, 1.0)

        assert jnp.allclose(C, C.T), "Collocation matrix should be symmetric"

    def test_diagonal_ones(self):
        """Diagonal should be 1 (r=0 -> phi=1)."""
        params = jnp.array([[1.0, 2.0, 3.0]])
        params_range = jnp.array([2.0])
        C = build_collocation_matrix(params, params_range, 1.0)

        assert jnp.allclose(jnp.diag(C), 1.0), "Diagonal elements should be 1"

    def test_shape(self):
        """Output shape should be (n_train, n_train)."""
        n_train = 7
        params = jnp.linspace(0, 10, n_train)[None, :]
        params_range = jnp.array([10.0])
        C = build_collocation_matrix(params, params_range, 0.5)

        assert C.shape == (n_train, n_train), f"Expected ({n_train}, {n_train}), got {C.shape}"

    def test_multi_param(self):
        """Should work with multiple parameters."""
        n_train = 5
        n_params = 3
        params = jnp.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ])
        params_range = jnp.array([4.0, 0.4, 40.0])
        C = build_collocation_matrix(params, params_range, 1.0)

        assert C.shape == (n_train, n_train)
        assert jnp.allclose(C, C.T), "Multi-param collocation should be symmetric"
        assert jnp.allclose(jnp.diag(C), 1.0), "Diagonal should still be 1"

    def test_values_positive(self):
        """All values should be positive (IMQ kernel is always positive)."""
        params = jnp.array([[1.0, 5.0, 10.0, 20.0]])
        params_range = jnp.array([19.0])
        C = build_collocation_matrix(params, params_range, 0.5)

        assert jnp.all(C > 0), "All values should be positive"

    def test_values_bounded(self):
        """Values should be in (0, 1] for IMQ kernel."""
        params = jnp.array([[1.0, 5.0, 10.0, 20.0]])
        params_range = jnp.array([19.0])
        C = build_collocation_matrix(params, params_range, 0.5)

        assert jnp.all(C > 0), "All values should be > 0"
        assert jnp.all(C <= 1.0), "All values should be <= 1"


class TestBuildInferenceMatrix:
    """Test RBF inference matrix construction."""

    def test_shape(self):
        """Output shape should be (n_inf, n_train)."""
        n_train = 5
        n_inf = 3
        train_params = jnp.linspace(0, 10, n_train)[None, :]
        inf_params = jnp.array([[2.5, 5.0, 7.5]])
        params_range = jnp.array([10.0])

        F = build_inference_matrix(train_params, inf_params, params_range, 1.0)

        assert F.shape == (n_inf, n_train), f"Expected ({n_inf}, {n_train}), got {F.shape}"

    def test_at_training_points(self):
        """Inference at training points should have max value 1 at that column."""
        train_params = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        inf_params = jnp.array([[3.0]])  # Exactly at training point
        params_range = jnp.array([4.0])

        F = build_inference_matrix(train_params, inf_params, params_range, 1.0)

        # At training point index 2, value should be 1
        assert jnp.isclose(F[0, 2], 1.0), f"Expected 1.0 at training point, got {F[0, 2]}"

    def test_values_positive(self):
        """All values should be positive."""
        train_params = jnp.array([[1.0, 5.0, 10.0]])
        inf_params = jnp.array([[2.0, 7.0]])
        params_range = jnp.array([9.0])

        F = build_inference_matrix(train_params, inf_params, params_range, 0.5)

        assert jnp.all(F > 0), "All values should be positive"

    def test_multi_param(self):
        """Should work with multiple parameters."""
        train_params = jnp.array([
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
        ])
        inf_params = jnp.array([
            [1.5, 2.5],
            [0.15, 0.25],
        ])
        params_range = jnp.array([2.0, 0.2])

        F = build_inference_matrix(train_params, inf_params, params_range, 1.0)

        assert F.shape == (2, 3)
        assert jnp.all(F > 0)
