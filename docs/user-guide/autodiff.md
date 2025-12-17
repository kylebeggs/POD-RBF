# Autodifferentiation

POD-RBF is built on JAX, enabling automatic differentiation through the inference functions. This is useful for optimization, sensitivity analysis, and inverse problems.

## Computing Gradients

Use `jax.grad` to compute gradients with respect to parameters:

```python
import jax
import jax.numpy as jnp
import pod_rbf

# Train model
result = pod_rbf.train(snapshot, params)
state = result.state

# Define an objective function
def objective(param):
    prediction = pod_rbf.inference_single(state, param)
    return jnp.sum(prediction ** 2)

# Compute gradient
grad_fn = jax.grad(objective)
gradient = grad_fn(jnp.array(450.0))
```

## Optimization Example

Find the parameter value that minimizes a cost function:

```python
import jax
import jax.numpy as jnp
from jax import grad

def cost_function(param, target):
    prediction = pod_rbf.inference_single(state, param)
    return jnp.mean((prediction - target) ** 2)

# Gradient descent
param = jnp.array(500.0)  # Initial guess
learning_rate = 10.0

for i in range(100):
    grad_val = grad(cost_function)(param, target_solution)
    param = param - learning_rate * grad_val

print(f"Optimal parameter: {param}")
```

## Inverse Problems

For inverse problems where you want to find the parameter that produced an observed solution:

```python
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

def inverse_objective(param):
    prediction = pod_rbf.inference_single(state, param)
    return jnp.sum((prediction - observed_solution) ** 2)

# Use BFGS optimization
result = minimize(
    inverse_objective,
    x0=jnp.array(500.0),
    method="BFGS",
)

recovered_param = result.x
```

## Sensitivity Analysis

Compute how sensitive the solution is to parameter changes:

```python
import jax
import jax.numpy as jnp

# Jacobian: how each output point changes with the parameter
jacobian_fn = jax.jacobian(
    lambda p: pod_rbf.inference_single(state, p)
)
sensitivity = jacobian_fn(jnp.array(450.0))

# sensitivity shape: (n_samples,) for single parameter
# Positive values indicate the solution increases with the parameter
```

## Multi-Parameter Gradients

For models with multiple parameters:

```python
def objective(params):
    # params: [Re, Ma]
    prediction = pod_rbf.inference_single(state, params)
    return jnp.sum(prediction ** 2)

# Gradient with respect to all parameters
grad_fn = jax.grad(objective)
gradients = grad_fn(jnp.array([450.0, 0.15]))
# gradients shape: (2,) - one gradient per parameter
```

## JIT Compilation

For performance, JIT-compile your gradient functions:

```python
@jax.jit
def compute_gradient(param):
    return jax.grad(objective)(param)

# First call compiles; subsequent calls are fast
gradient = compute_gradient(jnp.array(450.0))
```

## Higher-Order Derivatives

JAX supports higher-order derivatives:

```python
# Second derivative (Hessian for scalar output)
hessian_fn = jax.hessian(objective)
hessian = hessian_fn(jnp.array(450.0))
```
