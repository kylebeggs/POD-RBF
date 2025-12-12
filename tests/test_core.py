"""Tests for core train/inference functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pod_rbf
from pod_rbf.core import inference, inference_single, train
from pod_rbf.types import ModelState, TrainConfig, TrainResult


class TestTrain:
    """Test training function."""

    @pytest.fixture
    def linear_data(self):
        """Create simple linear test data: f(x, p) = p * x."""
        x = jnp.linspace(0, 1, 50)
        params = jnp.linspace(1, 10, 10)
        # snapshot[i, j] = params[j] * x[i]
        snapshot = x[:, None] * params[None, :]
        return snapshot, params, x

    def test_returns_train_result(self, linear_data):
        """Train should return TrainResult."""
        snapshot, params, _ = linear_data
        result = train(snapshot, params)

        assert isinstance(result, TrainResult)
        assert isinstance(result.state, ModelState)
        assert result.n_modes > 0
        assert isinstance(result.used_eig_decomp, bool)

    def test_model_state_shapes(self, linear_data):
        """Model state should have correct shapes."""
        snapshot, params, _ = linear_data
        result = train(snapshot, params)
        state = result.state

        n_samples, n_snapshots = snapshot.shape
        n_modes = result.n_modes

        assert state.basis.shape == (n_samples, n_modes)
        assert state.weights.shape == (n_modes, n_snapshots)
        assert state.train_params.shape == (1, n_snapshots)
        assert state.params_range.shape == (1,)

    def test_custom_config(self, linear_data):
        """Should accept custom config."""
        snapshot, params, _ = linear_data
        config = TrainConfig(energy_threshold=0.9)
        result = train(snapshot, params, config=config)

        assert result.state.truncated_energy >= 0.9

    def test_fixed_shape_factor(self, linear_data):
        """Should accept fixed shape factor."""
        snapshot, params, _ = linear_data
        result = train(snapshot, params, shape_factor=0.5)

        assert result.state.shape_factor == 0.5

    def test_multi_param(self):
        """Should work with multiple parameters."""
        x = jnp.linspace(0, 1, 30)
        p1 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p2 = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        params = jnp.stack([p1, p2], axis=0)

        # f(x, p1, p2) = p1 * x + p2
        snapshot = x[:, None] * p1[None, :] + p2[None, :]

        result = train(snapshot, params)

        assert result.state.train_params.shape == (2, 5)
        assert result.state.params_range.shape == (2,)


class TestInference:
    """Test inference functions."""

    @pytest.fixture
    def trained_model(self):
        """Create and train a model on linear data."""
        x = jnp.linspace(0, 1, 50)
        params = jnp.linspace(1, 10, 10)
        snapshot = x[:, None] * params[None, :]
        result = train(snapshot, params)
        return result.state, x, params, snapshot

    def test_inference_single_shape(self, trained_model):
        """inference_single should return 1D array."""
        state, x, _, _ = trained_model
        pred = inference_single(state, jnp.array(5.0))

        assert pred.shape == (len(x),)

    def test_inference_batch_shape(self, trained_model):
        """inference should return 2D array for batch input."""
        state, x, _, _ = trained_model
        pred = inference(state, jnp.array([[2.0, 5.0, 8.0]]))

        assert pred.shape == (len(x), 3)

    def test_interpolation_at_training_points(self, trained_model):
        """Inference at training points should match training data."""
        state, x, params, snapshot = trained_model

        for i, p in enumerate(params):
            pred = inference_single(state, p)
            expected = snapshot[:, i]
            assert jnp.allclose(pred, expected, rtol=1e-3), f"Mismatch at training point {i}"

    def test_interpolation_between_points(self, trained_model):
        """Inference between training points should be accurate for linear data."""
        state, x, _, _ = trained_model

        # Test at midpoint
        pred = inference_single(state, jnp.array(5.5))
        expected = 5.5 * x

        assert jnp.allclose(pred, expected, rtol=1e-2), "Interpolation at midpoint should be accurate"

    def test_scalar_param(self, trained_model):
        """Should handle scalar parameter input."""
        state, x, _, _ = trained_model
        pred = inference_single(state, 5.0)

        assert pred.shape == (len(x),)


class TestGradients:
    """Test autodifferentiation capabilities."""

    @pytest.fixture
    def trained_model(self):
        """Create trained model for gradient tests."""
        x = jnp.linspace(0, 1, 50)
        params = jnp.linspace(1, 10, 10)
        snapshot = x[:, None] * params[None, :]
        result = train(snapshot, params)
        return result.state, x

    def test_grad_wrt_param(self, trained_model):
        """Gradient of inference w.r.t. parameter should exist and be non-zero."""
        state, _ = trained_model

        def loss(p):
            pred = inference_single(state, p)
            return jnp.sum(pred**2)

        grad_fn = jax.grad(loss)
        grad = grad_fn(jnp.array(5.0))

        assert not jnp.isnan(grad), "Gradient should not be NaN"
        assert grad != 0.0, "Gradient should be non-zero"

    def test_inverse_problem(self, trained_model):
        """Test solving inverse problem via gradient descent."""
        state, x = trained_model

        # Target: parameter = 7.5
        target = 7.5 * x

        def loss(p):
            pred = inference_single(state, p)
            return jnp.mean((pred - target) ** 2)

        # Gradient descent
        p = jnp.array(5.0)  # Initial guess
        lr = 0.5

        for _ in range(50):
            g = jax.grad(loss)(p)
            p = p - lr * g

        assert jnp.abs(p - 7.5) < 0.1, f"Recovered parameter {p} should be close to 7.5"

    def test_jacobian(self, trained_model):
        """Jacobian should have correct shape."""
        state, x = trained_model

        jacobian = jax.jacobian(lambda p: inference_single(state, p))(jnp.array(5.0))

        assert jacobian.shape == (len(x),), f"Jacobian shape should be ({len(x)},), got {jacobian.shape}"

    def test_value_and_grad(self, trained_model):
        """value_and_grad should work."""
        state, _ = trained_model

        def loss(p):
            pred = inference_single(state, p)
            return jnp.sum(pred**2)

        val, grad = jax.value_and_grad(loss)(jnp.array(5.0))

        assert not jnp.isnan(val)
        assert not jnp.isnan(grad)


class TestSchurComplement:
    """Test Schur complement solver integration."""

    @pytest.fixture
    def linear_data(self):
        """Create simple linear test data: f(x, p) = p * x."""
        x = jnp.linspace(0, 1, 50)
        params = jnp.linspace(1, 10, 10)
        snapshot = x[:, None] * params[None, :]
        return snapshot, params, x

    def test_model_state_has_poly_fields(self, linear_data):
        """Model state should have polynomial coefficient fields."""
        snapshot, params, _ = linear_data
        result = train(snapshot, params)
        state = result.state

        assert hasattr(state, "poly_coeffs"), "ModelState should have poly_coeffs field"
        assert hasattr(state, "poly_degree"), "ModelState should have poly_degree field"
        assert state.poly_degree == 2, "Default poly_degree should be 2"
        assert state.poly_coeffs is not None, "poly_coeffs should not be None with default config"

    def test_poly_coeffs_shape(self, linear_data):
        """Polynomial coefficients should have correct shape."""
        snapshot, params, _ = linear_data
        result = train(snapshot, params)
        state = result.state

        n_modes = result.n_modes
        # For 1D params with degree 2: 3 polynomial terms [1, p, p²]
        n_poly = 3
        assert state.poly_coeffs.shape == (n_modes, n_poly), f"Expected ({n_modes}, {n_poly}), got {state.poly_coeffs.shape}"

    def test_poly_degree_0_fallback(self, linear_data):
        """poly_degree=0 should use pinv fallback."""
        snapshot, params, _ = linear_data
        config = TrainConfig(poly_degree=0)
        result = train(snapshot, params, config=config)
        state = result.state

        assert state.poly_degree == 0
        assert state.poly_coeffs is None, "poly_coeffs should be None when poly_degree=0"

    def test_interpolation_with_schur(self, linear_data):
        """Schur complement solver should interpolate training points accurately."""
        snapshot, params, x = linear_data
        result = train(snapshot, params)
        state = result.state

        for i, p in enumerate(params):
            pred = inference_single(state, p)
            expected = snapshot[:, i]
            assert jnp.allclose(pred, expected, rtol=1e-3), f"Mismatch at training point {i}"

    def test_interpolation_between_points_schur(self, linear_data):
        """Schur solver should interpolate accurately between training points."""
        snapshot, params, x = linear_data
        result = train(snapshot, params)
        state = result.state

        pred = inference_single(state, jnp.array(5.5))
        expected = 5.5 * x

        assert jnp.allclose(pred, expected, rtol=1e-2), "Interpolation at midpoint should be accurate"

    def test_grad_with_schur(self, linear_data):
        """Autodiff should work through Schur complement solver."""
        snapshot, params, _ = linear_data
        result = train(snapshot, params)
        state = result.state

        def loss(p):
            pred = inference_single(state, p)
            return jnp.sum(pred**2)

        grad_fn = jax.grad(loss)
        grad = grad_fn(jnp.array(5.0))

        assert not jnp.isnan(grad), "Gradient should not be NaN"
        assert grad != 0.0, "Gradient should be non-zero"

    def test_multi_param_with_schur(self):
        """Schur solver should work with multiple parameters."""
        x = jnp.linspace(0, 1, 30)
        # Use uncorrelated parameters to avoid rank-deficient polynomial basis
        p1 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p2 = jnp.array([0.5, 0.1, 0.4, 0.2, 0.3])
        params = jnp.stack([p1, p2], axis=0)

        snapshot = x[:, None] * p1[None, :] + p2[None, :]

        result = train(snapshot, params)
        state = result.state

        # For 2D params with degree 2: 6 polynomial terms
        n_poly = 6
        assert state.poly_coeffs.shape[1] == n_poly, f"Expected {n_poly} poly terms, got {state.poly_coeffs.shape[1]}"

        # Test interpolation at training points
        for i in range(len(p1)):
            pred = inference_single(state, jnp.array([p1[i], p2[i]]))
            expected = snapshot[:, i]
            assert jnp.allclose(pred, expected, rtol=1e-3), f"Mismatch at training point {i}"


class TestJIT:
    """Test JIT compilation."""

    @pytest.fixture
    def trained_model(self):
        """Create trained model for JIT tests."""
        x = jnp.linspace(0, 1, 50)
        params = jnp.linspace(1, 10, 10)
        snapshot = x[:, None] * params[None, :]
        result = train(snapshot, params)
        return result.state

    def test_inference_single_jit(self, trained_model):
        """inference_single should JIT compile via closure pattern."""
        state = trained_model

        # Create a closure that captures state (recommended pattern for JAX)
        @jax.jit
        def jitted_inference(p):
            return inference_single(state, p)

        # First call compiles
        pred1 = jitted_inference(jnp.array(5.0))
        # Second call uses cached compilation
        pred2 = jitted_inference(jnp.array(6.0))

        assert pred1.shape == (50,)
        assert pred2.shape == (50,)

    def test_inference_batch_jit(self, trained_model):
        """inference should JIT compile via closure pattern."""
        state = trained_model

        @jax.jit
        def jitted_inference(params):
            return inference(state, params)

        pred = jitted_inference(jnp.array([[2.0, 5.0, 8.0]]))

        assert pred.shape == (50, 3)

    def test_grad_jit(self, trained_model):
        """Gradient computation should JIT compile."""
        state = trained_model

        @jax.jit
        def loss_and_grad(p):
            pred = inference_single(state, p)
            loss_val = jnp.sum(pred**2)
            return loss_val

        grad_fn = jax.jit(jax.grad(loss_and_grad))

        loss = loss_and_grad(jnp.array(5.0))
        grad = grad_fn(jnp.array(5.0))

        assert not jnp.isnan(loss)
        assert not jnp.isnan(grad)
