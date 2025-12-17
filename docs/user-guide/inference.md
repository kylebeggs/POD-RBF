# Inference

After training, use the model to predict solutions at new parameter values.

## Single-Point Inference

For predicting at a single parameter value:

```python
import pod_rbf
import jax.numpy as jnp

# Train the model
result = pod_rbf.train(snapshot, params)

# Predict at a new parameter
prediction = pod_rbf.inference_single(result.state, jnp.array(450.0))
```

The output shape is `(n_samples,)` - the same as one column of your snapshot matrix.

## Batch Inference

For predicting at multiple parameter values simultaneously:

```python
# Predict at multiple parameters
new_params = jnp.array([350.0, 450.0, 550.0])
predictions = pod_rbf.inference(result.state, new_params)
```

The output shape is `(n_samples, n_points)` where `n_points` is the number of parameter values.

## Multi-Parameter Inference

For models trained with multiple parameters:

```python
# Single point with 2 parameters
param = jnp.array([450.0, 0.15])  # [Re, Ma]
prediction = pod_rbf.inference_single(result.state, param)

# Batch with 2 parameters
params = jnp.array([
    [350.0, 0.1],
    [450.0, 0.15],
    [550.0, 0.2],
])
predictions = pod_rbf.inference(result.state, params)
```

## Using a Saved Model

Load a previously saved model and use it for inference:

```python
state = pod_rbf.load_model("model.pkl")
prediction = pod_rbf.inference_single(state, jnp.array(450.0))
```

## Performance Tips

1. **Use batch inference** when predicting at multiple parameter values - it's more efficient than calling `inference_single` in a loop.

2. **JIT compilation** - The inference functions are JAX-compatible and can be JIT-compiled for faster repeated calls:

```python
import jax

inference_jit = jax.jit(lambda p: pod_rbf.inference_single(state, p))
prediction = inference_jit(jnp.array(450.0))
```

3. **GPU acceleration** - If JAX is configured with GPU support, inference will automatically use the GPU.
