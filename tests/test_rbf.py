"""Tests for RBF matrix construction."""

import jax.numpy as jnp
import pytest

from pod_rbf.rbf import (
    build_collocation_matrix,
    build_inference_matrix,
    build_polynomial_basis,
    solve_augmented_system_schur,
)


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


class TestBuildPolynomialBasis:
    """Test polynomial basis matrix construction."""

    def test_degree_0_shape(self):
        """Degree 0 should return constant column only."""
        params = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        params_range = jnp.array([4.0])
        P = build_polynomial_basis(params, params_range, degree=0)

        assert P.shape == (5, 1), f"Expected (5, 1), got {P.shape}"
        assert jnp.allclose(P[:, 0], 1.0), "Constant column should be all ones"

    def test_degree_1_shape(self):
        """Degree 1 should return constant + linear terms."""
        params = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        params_range = jnp.array([4.0])
        P = build_polynomial_basis(params, params_range, degree=1)

        # 1D: [1, p] -> 2 columns
        assert P.shape == (5, 2), f"Expected (5, 2), got {P.shape}"
        assert jnp.allclose(P[:, 0], 1.0), "First column should be all ones"

    def test_degree_2_shape_1d(self):
        """Degree 2 with 1 param should return 3 columns."""
        params = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        params_range = jnp.array([4.0])
        P = build_polynomial_basis(params, params_range, degree=2)

        # 1D: [1, p, p²] -> 3 columns
        assert P.shape == (5, 3), f"Expected (5, 3), got {P.shape}"

    def test_degree_2_shape_2d(self):
        """Degree 2 with 2 params should return 6 columns."""
        params = jnp.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        ])
        params_range = jnp.array([4.0, 0.4])
        P = build_polynomial_basis(params, params_range, degree=2)

        # 2D: [1, p1, p2, p1², p2², p1*p2] -> 6 columns
        assert P.shape == (5, 6), f"Expected (5, 6), got {P.shape}"

    def test_normalized_values(self):
        """Polynomial values should scale appropriately with params_range."""
        params = jnp.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
        params_range = jnp.array([40.0])
        P = build_polynomial_basis(params, params_range, degree=2)

        # Normalized values are params/range, so for [10, 20, 30, 40, 50]/40 = [0.25, 0.5, 0.75, 1.0, 1.25]
        # Constant term should be 1
        assert jnp.allclose(P[:, 0], 1.0), "Constant column should be all ones"
        # Linear terms should be params/range
        expected_linear = jnp.array([0.25, 0.5, 0.75, 1.0, 1.25])
        assert jnp.allclose(P[:, 1], expected_linear), f"Linear term mismatch: {P[:, 1]} vs {expected_linear}"

    def test_multi_param_cross_terms(self):
        """Cross terms should be computed correctly for multi-param case."""
        params = jnp.array([
            [0.0, 2.0, 4.0],
            [0.0, 2.0, 4.0],
        ])
        params_range = jnp.array([4.0, 4.0])
        P = build_polynomial_basis(params, params_range, degree=2)

        # Columns: [1, p1, p2, p1², p2², p1*p2]
        # Normalized: [0, 0.5, 1] for both params
        assert P.shape == (3, 6)
        # Last column is cross term p1*p2
        # At normalized values [0, 0.5, 1], cross terms are [0, 0.25, 1]
        expected_cross = jnp.array([0.0, 0.25, 1.0])
        assert jnp.allclose(P[:, 5], expected_cross), f"Cross term mismatch: {P[:, 5]} vs {expected_cross}"


class TestSolveAugmentedSystemSchur:
    """Test Schur complement solver."""

    def test_simple_system(self):
        """Solve a simple augmented system and verify solution."""
        # Simple 3x3 SPD matrix
        F = jnp.array([
            [4.0, 1.0, 0.5],
            [1.0, 3.0, 0.5],
            [0.5, 0.5, 2.0],
        ])
        # Linear polynomial basis
        P = jnp.array([
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
        ])
        # Single RHS
        rhs = jnp.array([[1.0, 2.0, 3.0]])

        rbf_weights, poly_coeffs = solve_augmented_system_schur(F, P, rhs)

        assert rbf_weights.shape == (1, 3), f"RBF weights shape: {rbf_weights.shape}"
        assert poly_coeffs.shape == (1, 2), f"Poly coeffs shape: {poly_coeffs.shape}"

        # Verify solution satisfies augmented system
        # F @ λ + P @ c = rhs
        lhs1 = F @ rbf_weights.T + P @ poly_coeffs.T
        assert jnp.allclose(lhs1.T, rhs, rtol=1e-5), f"First equation not satisfied: {lhs1.T} vs {rhs}"

        # P.T @ λ = 0 (orthogonality constraint)
        lhs2 = P.T @ rbf_weights.T
        assert jnp.allclose(lhs2, 0, atol=1e-10), f"Orthogonality constraint not satisfied: {lhs2}"

    def test_multiple_rhs(self):
        """Solve system with multiple right-hand sides."""
        F = jnp.array([
            [4.0, 1.0, 0.5],
            [1.0, 3.0, 0.5],
            [0.5, 0.5, 2.0],
        ])
        P = jnp.array([
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
        ])
        # Two RHS
        rhs = jnp.array([
            [1.0, 2.0, 3.0],
            [0.5, 1.0, 1.5],
        ])

        rbf_weights, poly_coeffs = solve_augmented_system_schur(F, P, rhs)

        assert rbf_weights.shape == (2, 3)
        assert poly_coeffs.shape == (2, 2)

        # Verify both solutions satisfy the system
        for i in range(2):
            lhs1 = F @ rbf_weights[i, :] + P @ poly_coeffs[i, :]
            assert jnp.allclose(lhs1, rhs[i, :], rtol=1e-5), f"RHS {i}: First equation not satisfied"

            lhs2 = P.T @ rbf_weights[i, :]
            assert jnp.allclose(lhs2, 0, atol=1e-10), f"RHS {i}: Orthogonality constraint not satisfied"

    def test_polynomial_reproduction(self):
        """Schur complement should reproduce polynomials exactly."""
        # If RHS is in the span of P, polynomial coeffs should capture it fully
        n_points = 5
        params = jnp.linspace(0, 1, n_points)[None, :]
        params_range = jnp.array([1.0])

        F = build_collocation_matrix(params, params_range, 0.5)
        P = build_polynomial_basis(params, params_range, degree=1)  # [1, p]

        # RHS that's a linear function: f = 2 + 3*p
        rhs = 2.0 + 3.0 * params  # shape (1, n_points)

        rbf_weights, poly_coeffs = solve_augmented_system_schur(F, P, rhs)

        # RBF weights should be near zero (polynomial captures everything)
        assert jnp.allclose(rbf_weights, 0, atol=1e-8), f"RBF weights should be ~0 for polynomial RHS: {rbf_weights}"

        # Poly coeffs should be [2, 3]
        assert jnp.allclose(poly_coeffs[0, 0], 2.0, rtol=1e-5), f"Constant coeff: {poly_coeffs[0, 0]}"
        assert jnp.allclose(poly_coeffs[0, 1], 3.0, rtol=1e-5), f"Linear coeff: {poly_coeffs[0, 1]}"
