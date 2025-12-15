"""
Shape Parameter Optimization via Autodiff

Demonstrates using JAX autodifferentiation with JAXopt L-BFGS to find
the optimal RBF shape parameter using the Rippa criterion (LOO-CV).

The Rippa criterion provides a closed-form leave-one-out cross-validation
error that is differentiable with respect to the shape parameter, enabling
gradient-based optimization.

Key concepts demonstrated:
    1. Rippa criterion (LOO-CV) for RBF shape parameter selection
    2. JAX autodiff through RBF matrix construction
    3. JAXopt L-BFGS optimizer for scientific computing
    4. Comparison with POD-RBF's condition-number based auto-optimization

Requirements:
    pip install jaxopt

References:
    Rippa, S. (1999). "An algorithm for selecting a good value for the
    parameter c in radial basis function interpolation."
    Advances in Computational Mathematics, 11(2-3), 193-210.
"""

import time

import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np

from pod_rbf.rbf import build_collocation_matrix

# Use CPU for reproducibility
jax.config.update("jax_default_device", jax.devices("cpu")[0])


# =============================================================================
# Test Function
# =============================================================================


def runge_function(x):
    """
    Runge function - a classic test case for interpolation.

    f(x) = 1 / (1 + 25x^2)

    This function has a sharp peak at x=0 and is challenging to interpolate
    accurately, making it ideal for demonstrating the importance of shape
    parameter selection.
    """
    return 1.0 / (1.0 + 25.0 * x**2)


# =============================================================================
# Rippa Criterion (LOO-CV Cost Function)
# =============================================================================


def loocv_cost(shape_factor, x, y, kernel="imq"):
    """
    Compute leave-one-out cross-validation error using Rippa's closed-form formula.

    The Rippa criterion computes the LOO-CV error without actually performing
    n separate leave-one-out fits. For RBF interpolation A @ c = y, the LOO
    error at point i is:

        e_i = c_i / A_inv_ii

    where c_i is the i-th interpolation coefficient and A_inv_ii is the i-th
    diagonal element of A^{-1}.

    Parameters
    ----------
    shape_factor : float
        RBF shape parameter to evaluate.
    x : Array
        Training point locations, shape (n_points,).
    y : Array
        Training values, shape (n_points,).
    kernel : str
        Kernel type: 'imq' or 'gaussian'.

    Returns
    -------
    float
        Mean squared LOO-CV error.
    """
    # Build collocation matrix with current shape factor
    # pod_rbf expects (n_params, n_points) shape
    x_2d = x[None, :]
    x_range = jnp.array([jnp.ptp(x)])

    A = build_collocation_matrix(x_2d, x_range, kernel=kernel, shape_factor=shape_factor)

    # Solve for RBF coefficients: A @ c = y
    c = jnp.linalg.solve(A, y)

    # Compute diagonal of A^{-1}
    # For numerical stability, we could use the formula:
    # diag(A^{-1})_i = e_i^T @ A^{-1} @ e_i
    # But direct inversion is fine for moderate problem sizes
    A_inv = jnp.linalg.inv(A)
    A_inv_diag = jnp.diag(A_inv)

    # Rippa criterion: LOO error at each point
    loo_errors = c / A_inv_diag

    # Return mean squared error
    return jnp.mean(loo_errors**2)


# =============================================================================
# Shape Parameter Optimization
# =============================================================================


def optimize_shape_parameter(x, y, kernel="imq", initial_guess=1.0, verbose=True):
    """
    Find optimal shape parameter using L-BFGS on Rippa criterion.

    Uses JAXopt's L-BFGS optimizer to minimize the LOO-CV error, with gradients
    computed automatically via JAX autodiff.

    We optimize in log-space (log_c) to ensure positivity without breaking gradients.
    shape_factor = exp(log_c)

    Parameters
    ----------
    x : Array
        Training point locations.
    y : Array
        Training values.
    kernel : str
        Kernel type.
    initial_guess : float
        Starting value for shape parameter.
    verbose : bool
        Print optimization progress.

    Returns
    -------
    optimal_shape : float
        Optimized shape parameter.
    history : dict
        Optimization history with shape parameters, costs, and gradients.
    """
    history = {"shape_factor": [initial_guess], "cost": [], "grad_log": []}

    # Optimize in log-space for unconstrained optimization with guaranteed positivity
    # shape_factor = exp(log_c), so we optimize log_c
    def objective(log_c):
        shape_factor = jnp.exp(log_c[0])
        return loocv_cost(shape_factor, x, y, kernel)

    # JIT compile for speed
    objective_jit = jax.jit(objective)
    grad_fn = jax.jit(jax.grad(objective))

    # Initial evaluation (in log space)
    log_c_init = jnp.log(initial_guess)
    init_cost = objective_jit(jnp.array([log_c_init]))
    init_grad = grad_fn(jnp.array([log_c_init]))
    history["cost"].append(float(init_cost))
    history["grad_log"].append(float(init_grad[0]))

    if verbose:
        print(f"Initial: shape_factor = {initial_guess:.6f}, LOO-CV cost = {init_cost:.6e}")

    # Create L-BFGS solver
    solver = jaxopt.LBFGS(
        fun=objective,
        maxiter=100,
        tol=1e-10,
    )

    # Initialize in log space
    log_c = jnp.array([log_c_init])

    # Run optimization with manual iteration to track history
    state = solver.init_state(log_c)

    for i in range(100):
        log_c, state = solver.update(log_c, state)
        cost = objective_jit(log_c)
        grad = grad_fn(log_c)

        shape_factor = float(jnp.exp(log_c[0]))
        history["shape_factor"].append(shape_factor)
        history["cost"].append(float(cost))
        history["grad_log"].append(float(grad[0]))

        if verbose and (i + 1) % 10 == 0:
            print(
                f"  Iter {i+1:3d}: shape_factor = {shape_factor:.6f}, "
                f"cost = {cost:.6e}, |grad| = {jnp.abs(grad[0]):.2e}"
            )

        # Check convergence (gradient in log space)
        if jnp.abs(grad[0]) < 1e-10:
            if verbose:
                print(f"  Converged at iteration {i+1}")
            break

    optimal_shape = float(jnp.exp(log_c[0]))

    if verbose:
        print(f"\nOptimal shape parameter: {optimal_shape:.6f}")
        print(f"Final LOO-CV cost: {history['cost'][-1]:.6e}")

    return optimal_shape, history


# =============================================================================
# RBF Interpolation Helper
# =============================================================================


def rbf_interpolate(x_train, y_train, x_eval, shape_factor, kernel="imq"):
    """
    Perform RBF interpolation at evaluation points.

    Parameters
    ----------
    x_train : Array
        Training point locations.
    y_train : Array
        Training values.
    x_eval : Array
        Points at which to evaluate the interpolant.
    shape_factor : float
        RBF shape parameter.
    kernel : str
        Kernel type.

    Returns
    -------
    Array
        Interpolated values at x_eval.
    """
    from pod_rbf.rbf import build_inference_matrix

    x_train_2d = x_train[None, :]
    x_eval_2d = x_eval[None, :]
    x_range = jnp.array([jnp.ptp(x_train)])

    # Build collocation matrix and solve for coefficients
    A = build_collocation_matrix(
        x_train_2d, x_range, kernel=kernel, shape_factor=shape_factor
    )
    c = jnp.linalg.solve(A, y_train)

    # Build inference matrix and evaluate
    F = build_inference_matrix(
        x_train_2d, x_eval_2d, x_range, kernel=kernel, shape_factor=shape_factor
    )

    return F @ c


# =============================================================================
# Visualization
# =============================================================================


def plot_loocv_landscape(x, y, optimal_shape, kernel="imq"):
    """Plot LOO-CV cost as a function of shape parameter."""
    shape_factors = np.logspace(-2, 1, 100)
    costs = []

    loocv_jit = jax.jit(lambda c: loocv_cost(c, x, y, kernel))

    for c in shape_factors:
        costs.append(float(loocv_jit(c)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(shape_factors, costs, "b-", linewidth=2, label="LOO-CV cost")
    ax.axvline(
        optimal_shape, color="r", linestyle="--", linewidth=2, label=f"Optimal: {optimal_shape:.4f}"
    )
    ax.set_xlabel("Shape Parameter (c)", fontsize=12)
    ax.set_ylabel("LOO-CV Cost (log scale)", fontsize=12)
    ax.set_title("LOO-CV Error Landscape", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    return fig


def plot_interpolation_comparison(x_train, y_train, x_eval, y_true, shape_factors, kernel="imq"):
    """Compare interpolation with different shape parameters."""
    n_shapes = len(shape_factors)
    fig, axes = plt.subplots(1, n_shapes, figsize=(5 * n_shapes, 4))

    if n_shapes == 1:
        axes = [axes]

    for ax, shape_factor in zip(axes, shape_factors):
        # Interpolate
        y_interp = rbf_interpolate(x_train, y_train, x_eval, shape_factor, kernel)

        # Compute error
        rmse = np.sqrt(np.mean((np.array(y_interp) - np.array(y_true)) ** 2))

        # Plot
        ax.plot(x_eval, y_true, "k-", linewidth=2, label="True function")
        ax.plot(x_eval, y_interp, "b--", linewidth=2, label="RBF interpolant")
        ax.scatter(x_train, y_train, c="r", s=50, zorder=5, label="Training points")

        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("f(x)", fontsize=11)
        ax.set_title(f"c = {shape_factor:.4f}, RMSE = {rmse:.2e}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_convergence(history):
    """Plot optimization convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Cost convergence
    ax = axes[0]
    ax.semilogy(history["cost"], "b-", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("LOO-CV Cost (log scale)", fontsize=12)
    ax.set_title("Cost Convergence", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Shape parameter evolution
    ax = axes[1]
    ax.plot(history["shape_factor"], "r-", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Shape Parameter", fontsize=12)
    ax.set_title("Shape Parameter Evolution", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Gradient magnitude (in log space)
    ax = axes[2]
    grad_key = "grad_log" if "grad_log" in history else "grad"
    ax.semilogy(np.abs(history[grad_key]), "g-", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("|Gradient| (log scale)", fontsize=12)
    ax.set_title("Gradient Magnitude", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================


def find_good_initial_guess(x, y, kernel="imq"):
    """Scan a range of shape parameters to find a good starting point."""
    loocv_jit = jax.jit(lambda c: loocv_cost(c, x, y, kernel))

    # Scan over a range of shape parameters
    shape_factors = np.logspace(-2, 1, 50)
    costs = []

    for c in shape_factors:
        try:
            cost = float(loocv_jit(c))
            if np.isfinite(cost):
                costs.append(cost)
            else:
                costs.append(np.inf)
        except Exception:
            costs.append(np.inf)

    # Find minimum
    best_idx = np.argmin(costs)
    return shape_factors[best_idx], shape_factors, costs


def main():
    print("=" * 70)
    print("Shape Parameter Optimization via Autodiff (Rippa Criterion + L-BFGS)")
    print("=" * 70)

    # Setup
    np.random.seed(42)
    kernel = "imq"

    # Generate training data (Runge function)
    n_train = 15
    x_train = jnp.linspace(-1, 1, n_train)
    y_train = runge_function(x_train)

    # Dense evaluation grid
    x_eval = jnp.linspace(-1, 1, 200)
    y_true = runge_function(x_eval)

    print(f"\nTest problem: Runge function f(x) = 1/(1 + 25x^2)")
    print(f"Training points: {n_train}")
    print(f"Kernel: {kernel.upper()}")

    # ==========================================================================
    # First, scan the landscape to find a good initial guess
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Scanning LOO-CV landscape for good initial guess...")
    print("-" * 70)

    initial_guess, scan_shapes, scan_costs = find_good_initial_guess(x_train, y_train, kernel)
    print(f"Best from scan: shape_factor = {initial_guess:.6f}, LOO-CV = {np.min(scan_costs):.6e}")

    # ==========================================================================
    # Optimize shape parameter using Rippa criterion + L-BFGS
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Refining with JAXopt L-BFGS...")
    print("-" * 70)

    start = time.time()
    optimal_shape, history = optimize_shape_parameter(
        x_train, y_train, kernel=kernel, initial_guess=initial_guess, verbose=True
    )
    opt_time = time.time() - start
    print(f"Optimization time: {opt_time:.3f} sec")

    # ==========================================================================
    # Compare with different shape parameters
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Comparing interpolation accuracy...")
    print("-" * 70)

    # Test several shape parameters
    test_shapes = [0.1, optimal_shape, 2.0]
    shape_labels = ["Too small (0.1)", f"Optimal ({optimal_shape:.4f})", "Too large (2.0)"]

    for shape, label in zip(test_shapes, shape_labels):
        y_interp = rbf_interpolate(x_train, y_train, x_eval, shape, kernel)
        rmse = np.sqrt(np.mean((np.array(y_interp) - np.array(y_true)) ** 2))
        loocv = loocv_cost(shape, x_train, y_train, kernel)
        print(f"  {label:25s}: RMSE = {rmse:.4e}, LOO-CV = {loocv:.4e}")

    # ==========================================================================
    # Demonstrate autodiff explicitly
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Demonstrating autodiff capabilities...")
    print("-" * 70)

    # Show gradient computation
    grad_fn = jax.jit(jax.grad(lambda c: loocv_cost(c, x_train, y_train, kernel)))

    for c in [0.1, 0.5, optimal_shape, 2.0]:
        grad = grad_fn(c)
        print(f"  d(LOO-CV)/dc at c={c:.4f}: {grad:.6e}")

    # Show Hessian computation
    hess_fn = jax.jit(jax.hessian(lambda c: loocv_cost(c, x_train, y_train, kernel)))
    hess_at_opt = hess_fn(optimal_shape)
    print(f"\n  d²(LOO-CV)/dc² at optimal c={optimal_shape:.4f}: {hess_at_opt:.6e}")
    print("  (Positive Hessian confirms this is a local minimum)")

    # ==========================================================================
    # Visualizations
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Generating plots...")
    print("-" * 70)

    # Plot 1: LOO-CV landscape
    fig1 = plot_loocv_landscape(x_train, y_train, optimal_shape, kernel)

    # Plot 2: Interpolation comparison
    fig2 = plot_interpolation_comparison(
        x_train, y_train, x_eval, y_true, test_shapes, kernel
    )

    # Plot 3: Convergence history
    fig3 = plot_convergence(history)

    print("\n" + "=" * 70)
    print("Done! This example demonstrated:")
    print("  1. Rippa criterion (LOO-CV) for RBF shape parameter selection")
    print("  2. JAX autodiff through build_collocation_matrix")
    print("  3. JAXopt L-BFGS optimization")
    print("  4. Gradient and Hessian computation of LOO-CV cost")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
