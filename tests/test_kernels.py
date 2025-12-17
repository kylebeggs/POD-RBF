"""
Tests for RBF kernel functions.
"""

import jax.numpy as jnp
import pytest

from pod_rbf.kernels import (
    KernelType,
    apply_kernel,
    kernel_gaussian,
    kernel_imq,
    kernel_polyharmonic_spline,
)


class TestKernelIMQ:
    """Test Inverse Multiquadrics kernel."""

    def test_imq_at_zero(self):
        """IMQ kernel should be 1 at r=0."""
        r2 = jnp.array([0.0])
        result = kernel_imq(r2, shape_factor=1.0)
        assert jnp.allclose(result, 1.0)

    def test_imq_bounded(self):
        """IMQ kernel values should be in (0, 1]."""
        r2 = jnp.array([0.0, 0.1, 1.0, 10.0, 100.0])
        result = kernel_imq(r2, shape_factor=1.0)
        assert jnp.all(result > 0)
        assert jnp.all(result <= 1.0)

    def test_imq_decay(self):
        """IMQ should decay monotonically with distance."""
        r2 = jnp.array([0.0, 1.0, 4.0, 9.0])
        result = kernel_imq(r2, shape_factor=1.0)
        # Check monotonic decrease
        assert jnp.all(result[:-1] >= result[1:])

    def test_imq_shape_factor_effect(self):
        """Larger shape factor should give slower decay."""
        r2 = jnp.array([4.0])
        val_small_c = kernel_imq(r2, shape_factor=0.5)
        val_large_c = kernel_imq(r2, shape_factor=2.0)
        # Larger c means slower decay, so value should be higher
        assert val_large_c > val_small_c


class TestKernelGaussian:
    """Test Gaussian kernel."""

    def test_gaussian_at_zero(self):
        """Gaussian kernel should be 1 at r=0."""
        r2 = jnp.array([0.0])
        result = kernel_gaussian(r2, shape_factor=1.0)
        assert jnp.allclose(result, 1.0)

    def test_gaussian_bounded(self):
        """Gaussian kernel values should be in (0, 1]."""
        r2 = jnp.array([0.0, 0.1, 1.0, 10.0, 100.0])
        result = kernel_gaussian(r2, shape_factor=1.0)
        assert jnp.all(result > 0)
        assert jnp.all(result <= 1.0)

    def test_gaussian_decay(self):
        """Gaussian should decay monotonically with distance."""
        r2 = jnp.array([0.0, 1.0, 4.0, 9.0])
        result = kernel_gaussian(r2, shape_factor=1.0)
        # Check monotonic decrease
        assert jnp.all(result[:-1] >= result[1:])

    def test_gaussian_exponential_decay(self):
        """Gaussian should decay exponentially."""
        r2 = jnp.array([0.0, 1.0])
        c = 1.0
        result = kernel_gaussian(r2, shape_factor=c)
        # At r²=1, should be exp(-1/c²) = exp(-1)
        assert jnp.allclose(result[1], jnp.exp(-1.0))

    def test_gaussian_shape_factor_effect(self):
        """Larger shape factor should give slower decay."""
        r2 = jnp.array([4.0])
        val_small_c = kernel_gaussian(r2, shape_factor=0.5)
        val_large_c = kernel_gaussian(r2, shape_factor=2.0)
        # Larger c means slower decay, so value should be higher
        assert val_large_c > val_small_c


class TestKernelPolyharmonicSpline:
    """Test Polyharmonic Spline kernels."""

    def test_phs_order_1(self):
        """PHS order 1: phi(r) = r."""
        r2 = jnp.array([0.0, 1.0, 4.0, 9.0])
        result = kernel_polyharmonic_spline(r2, order=1)
        expected = jnp.sqrt(r2)
        assert jnp.allclose(result, expected)

    def test_phs_order_3(self):
        """PHS order 3: phi(r) = r³."""
        r2 = jnp.array([0.0, 1.0, 4.0, 9.0])
        result = kernel_polyharmonic_spline(r2, order=3)
        r = jnp.sqrt(r2)
        expected = r**3
        assert jnp.allclose(result, expected)

    def test_phs_order_5(self):
        """PHS order 5: phi(r) = r⁵."""
        r2 = jnp.array([0.0, 1.0, 4.0, 9.0])
        result = kernel_polyharmonic_spline(r2, order=5)
        r = jnp.sqrt(r2)
        expected = r**5
        assert jnp.allclose(result, expected)

    def test_phs_order_2(self):
        """PHS order 2: phi(r) = r²*log(r)."""
        # Avoid r=0 for log
        r2 = jnp.array([1.0, 4.0, 9.0])
        result = kernel_polyharmonic_spline(r2, order=2)
        r = jnp.sqrt(r2)
        expected = (r**2) * jnp.log(r)
        assert jnp.allclose(result, expected)

    def test_phs_order_4(self):
        """PHS order 4: phi(r) = r⁴*log(r)."""
        # Avoid r=0 for log
        r2 = jnp.array([1.0, 4.0, 9.0])
        result = kernel_polyharmonic_spline(r2, order=4)
        r = jnp.sqrt(r2)
        expected = (r**4) * jnp.log(r)
        assert jnp.allclose(result, expected)

    def test_phs_at_zero_odd(self):
        """PHS odd orders should be 0 at r=0."""
        r2 = jnp.array([0.0])
        for order in [1, 3, 5]:
            result = kernel_polyharmonic_spline(r2, order=order)
            assert jnp.allclose(result, 0.0)

    def test_phs_at_zero_even(self):
        """PHS even orders should be 0 at r=0 (handled by jnp.where)."""
        r2 = jnp.array([0.0])
        for order in [2, 4]:
            result = kernel_polyharmonic_spline(r2, order=order)
            # Should be set to 0 by the where clause to avoid log(0)
            assert jnp.isfinite(result[0])
            assert jnp.allclose(result, 0.0)

    def test_phs_growth(self):
        """PHS should grow with distance (unlike IMQ/Gaussian)."""
        r2 = jnp.array([0.0, 1.0, 4.0])
        result = kernel_polyharmonic_spline(r2, order=3)
        # Should increase (not decay)
        assert result[0] <= result[1] <= result[2]


class TestApplyKernel:
    """Test kernel dispatcher."""

    def test_apply_imq(self):
        """Dispatcher should correctly apply IMQ kernel."""
        r2 = jnp.array([0.0, 1.0, 4.0])
        result = apply_kernel(r2, "imq", shape_factor=1.0, kernel_order=3)
        expected = kernel_imq(r2, shape_factor=1.0)
        assert jnp.allclose(result, expected)

    def test_apply_gaussian(self):
        """Dispatcher should correctly apply Gaussian kernel."""
        r2 = jnp.array([0.0, 1.0, 4.0])
        result = apply_kernel(r2, "gaussian", shape_factor=1.0, kernel_order=3)
        expected = kernel_gaussian(r2, shape_factor=1.0)
        assert jnp.allclose(result, expected)

    def test_apply_phs(self):
        """Dispatcher should correctly apply PHS kernel."""
        r2 = jnp.array([0.0, 1.0, 4.0])
        result = apply_kernel(r2, "polyharmonic_spline", shape_factor=None, kernel_order=3)
        expected = kernel_polyharmonic_spline(r2, order=3)
        assert jnp.allclose(result, expected)

    def test_apply_invalid_kernel(self):
        """Dispatcher should raise error for invalid kernel."""
        r2 = jnp.array([1.0])
        with pytest.raises(ValueError, match="is not a valid KernelType"):
            apply_kernel(r2, "invalid_kernel", shape_factor=1.0, kernel_order=3)


class TestKernelType:
    """Test KernelType enum."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert KernelType.IMQ == "imq"
        assert KernelType.GAUSSIAN == "gaussian"
        assert KernelType.POLYHARMONIC_SPLINE == "polyharmonic_spline"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert KernelType("imq") == KernelType.IMQ
        assert KernelType("gaussian") == KernelType.GAUSSIAN
        assert KernelType("polyharmonic_spline") == KernelType.POLYHARMONIC_SPLINE
