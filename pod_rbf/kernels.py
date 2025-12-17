"""
RBF kernel functions and dispatcher.

Supports multiple kernel types:
- Inverse Multi-Quadrics (IMQ): phi(r) = 1 / sqrt(r²/c² + 1)
- Gaussian: phi(r) = exp(-r²/c²)
- Polyharmonic Splines (PHS): phi(r) = r^k or r^k*log(r)
"""

from enum import Enum

import jax.numpy as jnp
from jax import Array


class KernelType(str, Enum):
    """RBF kernel types (internal use only)."""

    IMQ = "imq"
    GAUSSIAN = "gaussian"
    POLYHARMONIC_SPLINE = "polyharmonic_spline"


def kernel_imq(r2: Array, shape_factor: float) -> Array:
    """
    Inverse Multiquadrics kernel.

    phi(r) = 1 / sqrt(r²/c² + 1)

    Parameters
    ----------
    r2 : Array
        Squared distances between points.
    shape_factor : float
        Shape parameter c.

    Returns
    -------
    Array
        Kernel values, same shape as r2.
    """
    return 1.0 / jnp.sqrt(r2 / (shape_factor**2) + 1.0)


def kernel_gaussian(r2: Array, shape_factor: float) -> Array:
    """
    Gaussian kernel.

    phi(r) = exp(-r²/c²)

    Parameters
    ----------
    r2 : Array
        Squared distances between points.
    shape_factor : float
        Shape parameter c.

    Returns
    -------
    Array
        Kernel values, same shape as r2.
    """
    return jnp.exp(-r2 / (shape_factor**2))


def kernel_polyharmonic_spline(r2: Array, order: int) -> Array:
    """
    Polyharmonic spline kernel.

    - Odd order k: phi(r) = r^k
    - Even order k: phi(r) = r^k * log(r)

    Parameters
    ----------
    r2 : Array
        Squared distances between points.
    order : int
        Polynomial order (typically 1-5).
        Odd: r, r³, r⁵
        Even: r²log(r), r⁴log(r)

    Returns
    -------
    Array
        Kernel values, same shape as r2.

    Notes
    -----
    For even orders, handles r=0 case to avoid log(0) singularity.
    Uses r2-based formulation to avoid gradient singularity from sqrt at r2=0.
    """
    if order % 2 == 1:
        # Odd order: r^k = (r2)^(k/2)
        # Using power of r2 directly avoids sqrt gradient singularity at r2=0
        return jnp.power(r2, order / 2.0)
    else:
        # Even order: r^k * log(r) = (r2)^(k/2) * log(sqrt(r2))
        #           = (r2)^(k/2) * (1/2) * log(r2)
        # Handle r2=0 case: set to 0 when r2 < threshold
        return jnp.where(
            r2 > 1e-30,
            jnp.power(r2, order / 2.0) * 0.5 * jnp.log(r2),
            0.0,
        )


def apply_kernel(
    r2: Array,
    kernel: str,
    shape_factor: float | None,
    kernel_order: int,
) -> Array:
    """
    Apply RBF kernel to distance matrix.

    Dispatcher function that selects and applies the appropriate kernel
    based on the kernel type string.

    Parameters
    ----------
    r2 : Array
        Squared distances between points.
    kernel : str
        Kernel type: 'imq', 'gaussian', or 'polyharmonic_spline'.
    shape_factor : float | None
        Shape parameter for IMQ and Gaussian kernels.
        Ignored for polyharmonic splines.
    kernel_order : int
        Order for polyharmonic splines.
        Ignored for other kernels.

    Returns
    -------
    Array
        Kernel values applied to distance matrix.

    Raises
    ------
    ValueError
        If kernel type is not recognized.
    """
    kernel_type = KernelType(kernel)

    if kernel_type == KernelType.IMQ:
        return kernel_imq(r2, shape_factor)
    elif kernel_type == KernelType.GAUSSIAN:
        return kernel_gaussian(r2, shape_factor)
    elif kernel_type == KernelType.POLYHARMONIC_SPLINE:
        return kernel_polyharmonic_spline(r2, kernel_order)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")


# Kernel-specific defaults for shape parameter optimization
KERNEL_SHAPE_DEFAULTS = {
    "imq": {"c_low": 0.011, "c_high": 1.0, "c_step": 0.01},
    "gaussian": {"c_low": 0.1, "c_high": 10.0, "c_step": 0.1},
}
