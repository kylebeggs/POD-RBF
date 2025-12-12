"""
Design Optimization with Autodiff

Demonstrates using JAX autodifferentiation through POD-RBF for design optimization.

Problem: 1D nonlinear heat conduction with temperature-dependent thermal conductivity
    k(T) = k₀(1 + βT)

The nonlinear dependence on boundary temperature T_L creates a non-trivial optimization
landscape requiring multiple POD modes to capture.

Objective: Find boundary temperature T_L that achieves a target average temperature.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import pod_rbf


jax.config.update("jax_default_device", jax.devices("cpu")[0])

# Physical parameters
T_0 = 300.0  # Left boundary temperature (K)
BETA = 0.002  # Temperature coefficient for conductivity
L = 1.0  # Domain length


def analytical_solution(x, T_L):
    """
    Analytical solution for 1D steady heat conduction with k(T) = k₀(1 + βT).

    Uses Kirchhoff transformation: θ = T + (β/2)T²
    The transformed variable θ satisfies linear diffusion, giving θ(x) linear in x.
    Inverting the quadratic yields T(x).
    """
    theta_0 = T_0 + (BETA / 2) * T_0**2
    theta_L = T_L + (BETA / 2) * T_L**2
    theta = theta_0 + (theta_L - theta_0) * (x / L)
    return (-1 + np.sqrt(1 + 2 * BETA * theta)) / BETA


def build_snapshot_matrix(T_L_values, x):
    """Build snapshot matrix from analytical solutions at different T_L values."""
    print("Building snapshot matrix... ", end="")
    start = time.time()

    n_points = len(x)
    n_snapshots = len(T_L_values)
    snapshot = np.zeros((n_points, n_snapshots))

    for i, T_L in enumerate(T_L_values):
        snapshot[:, i] = analytical_solution(x, T_L)

    print(f"took {time.time() - start:.3f} sec")
    return snapshot


def run_optimization():
    # Spatial discretization
    n_points = 100
    x = np.linspace(0, L, n_points)

    # Training: sample T_L over range [350, 600] K
    T_L_train = np.linspace(350, 600, num=20)
    snapshot = build_snapshot_matrix(T_L_train, x)

    # Train ROM
    config = pod_rbf.TrainConfig(energy_threshold=0.9999, poly_degree=2)
    result = pod_rbf.train(snapshot, T_L_train, config)
    state = result.state

    print(f"Trained with {result.n_modes} modes, energy retained: {state.truncated_energy:.6f}")
    print(f"Cumulative energy per mode: {state.cumul_energy}")

    # Target: achieve average temperature of 400 K
    T_target = 400.0

    def objective(T_L):
        """Squared error between predicted average temp and target."""
        pred = pod_rbf.inference_single(state, T_L)
        avg_temp = jnp.mean(pred)
        return (avg_temp - T_target) ** 2

    # JIT-compile functions
    grad_fn = jax.jit(jax.grad(objective))
    obj_fn = jax.jit(objective)

    # Also track average temperature
    @jax.jit
    def avg_temp_fn(T_L):
        return jnp.mean(pod_rbf.inference_single(state, T_L))

    # Optimization settings
    T_L_init = jnp.array(550.0)  # Start far from optimal
    T_L = T_L_init
    lr = 1.0  # Learning rate
    n_iters = 30
    T_L_min, T_L_max = 350.0, 600.0  # Valid parameter range

    # Track history
    history = {
        "T_L": [float(T_L)],
        "objective": [float(obj_fn(T_L))],
        "avg_temp": [float(avg_temp_fn(T_L))],
        "grad": [],
    }

    print(f"\nOptimizing: find T_L to achieve average temperature = {T_target} K")
    print(f"Initial: T_L = {T_L_init:.1f} K, avg_temp = {history['avg_temp'][0]:.2f} K")

    # Gradient descent
    for i in range(n_iters):
        grad = grad_fn(T_L)
        history["grad"].append(float(grad))

        T_L = T_L - lr * grad
        T_L = jnp.clip(T_L, T_L_min, T_L_max)

        obj_val = obj_fn(T_L)
        avg_temp = avg_temp_fn(T_L)
        history["T_L"].append(float(T_L))
        history["objective"].append(float(obj_val))
        history["avg_temp"].append(float(avg_temp))

        if (i + 1) % 5 == 0:
            print(
                f"  Iter {i+1:3d}: T_L = {T_L:.2f} K, "
                f"avg_temp = {avg_temp:.2f} K, loss = {obj_val:.4e}"
            )

    T_L_opt = float(T_L)
    print(f"\nOptimal T_L = {T_L_opt:.2f} K (avg_temp = {history['avg_temp'][-1]:.2f} K)")

    # Get solutions for visualization
    pred_init = pod_rbf.inference_single(state, T_L_init)
    pred_opt = pod_rbf.inference_single(state, jnp.array(T_L_opt))

    # Analytical solutions for comparison
    T_analytical_init = analytical_solution(x, float(T_L_init))
    T_analytical_opt = analytical_solution(x, T_L_opt)

    return (
        history,
        x,
        pred_init,
        pred_opt,
        T_analytical_init,
        T_analytical_opt,
        T_L_init,
        T_L_opt,
        T_target,
    )


def plot_results(
    history, x, pred_init, pred_opt, T_anal_init, T_anal_opt, T_L_init, T_L_opt, T_target
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Objective convergence
    ax = axes[0, 0]
    ax.semilogy(history["objective"], "b-", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective (squared error)")
    ax.set_title("Optimization Convergence")
    ax.grid(True, alpha=0.3)

    # Average temperature convergence
    ax = axes[0, 1]
    ax.plot(history["avg_temp"], "r-", linewidth=2, label="Predicted avg temp")
    ax.axhline(y=T_target, color="g", linestyle="--", linewidth=2, label=f"Target: {T_target} K")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Temperature (K)")
    ax.set_title("Average Temperature vs Target")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Initial temperature profile
    ax = axes[1, 0]
    ax.plot(x, pred_init, "b-", linewidth=2, label="ROM prediction")
    ax.plot(x, T_anal_init, "r--", linewidth=2, label="Analytical")
    ax.axhline(
        y=float(jnp.mean(pred_init)),
        color="g",
        linestyle=":",
        label=f"Avg: {float(jnp.mean(pred_init)):.1f} K",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"Initial: T_L = {float(T_L_init):.1f} K")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Optimized temperature profile
    ax = axes[1, 1]
    ax.plot(x, pred_opt, "b-", linewidth=2, label="ROM prediction")
    ax.plot(x, T_anal_opt, "r--", linewidth=2, label="Analytical")
    ax.axhline(
        y=float(jnp.mean(pred_opt)),
        color="g",
        linestyle=":",
        label=f"Avg: {float(jnp.mean(pred_opt)):.1f} K",
    )
    ax.axhline(y=T_target, color="orange", linestyle="--", alpha=0.7, label=f"Target: {T_target} K")
    ax.set_xlabel("x")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"Optimized: T_L = {T_L_opt:.1f} K")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_optimization()
    plot_results(*results)
