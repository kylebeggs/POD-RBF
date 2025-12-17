# Saving & Loading

## Loading Snapshot Data

### From CSV Files

Load snapshots from a directory of CSV files:

```python
import pod_rbf

snapshot = pod_rbf.build_snapshot_matrix("path/to/data/")
```

By default, this:

- Loads all CSV files from the directory in alphanumeric order
- Skips the first row (header)
- Uses the first column

Customize with optional parameters:

```python
snapshot = pod_rbf.build_snapshot_matrix(
    "path/to/data/",
    skiprows=1,      # Skip first N rows (default: 1 for header)
    usecols=0,       # Column index to use (default: 0)
    verbose=True,    # Show progress bar (default: True)
)
```

### From NumPy Arrays

If your data is already in memory:

```python
import numpy as np

# Combine individual solutions into a snapshot matrix
# Each column is one snapshot
snapshot = np.column_stack([sol1, sol2, sol3, sol4, sol5])
```

## Saving Models

Save a trained model to disk:

```python
import pod_rbf

result = pod_rbf.train(snapshot, params)
pod_rbf.save_model("model.pkl", result.state)
```

The model is saved as a pickle file containing the `ModelState` NamedTuple.

## Loading Models

Load a previously saved model:

```python
state = pod_rbf.load_model("model.pkl")

# Use for inference
prediction = pod_rbf.inference_single(state, jnp.array(450.0))
```

## Model State Contents

The saved `ModelState` contains everything needed for inference:

| Field | Description |
|-------|-------------|
| `basis` | Truncated POD basis matrix |
| `weights` | RBF interpolation weights |
| `shape_factor` | Optimized RBF shape parameter |
| `train_params` | Training parameter values |
| `params_range` | Parameter ranges for normalization |
| `truncated_energy` | Energy retained after truncation |
| `cumul_energy` | Cumulative energy per mode |
| `poly_coeffs` | Polynomial coefficients (if used) |
| `poly_degree` | Polynomial degree used |
| `kernel` | Kernel type used |
| `kernel_order` | PHS order (for polyharmonic splines) |

## File Format Notes

- Models are saved using Python's `pickle` module
- Files are portable across machines with the same Python/JAX versions
- File size depends on the number of modes and training points
