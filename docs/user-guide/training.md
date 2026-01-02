# Training Models

## Building the Snapshot Matrix

The snapshot matrix contains your training data. Each column is a solution snapshot at a specific parameter value.

### From CSV Files

If your snapshots are stored as individual CSV files in a directory:

```python
import pod_rbf

snapshot = pod_rbf.build_snapshot_matrix("path/to/data/")
```

Files are loaded in alphanumeric order. The function expects one value per row in each CSV file.

!!! tip "File Organization"
    Keep all training snapshots in a dedicated directory. Files are sorted alphanumerically, so use consistent naming like `snapshot_001.csv`, `snapshot_002.csv`, etc.

### From Arrays

If you already have your data in memory:

```python
import numpy as np

# Shape: (n_samples, n_snapshots)
snapshot = np.column_stack([solution1, solution2, solution3, ...])
```

## Training

### Basic Training

```python
import pod_rbf
import numpy as np

params = np.array([100, 200, 300, 400, 500])
result = pod_rbf.train(snapshot, params)
```

The `result` object contains:

- `result.state` - The trained model state (use this for inference)
- `result.n_modes` - Number of POD modes retained
- `result.used_eig_decomp` - Whether eigendecomposition was used (vs SVD)

### Training Configuration

Customize training with `TrainConfig`:

```python
from pod_rbf import TrainConfig

config = TrainConfig(
    energy_threshold=0.99,  # Keep modes until 99% energy retained
    kernel="imq",           # RBF kernel: 'imq', 'gaussian', 'polyharmonic_spline'
    poly_degree=2,          # Polynomial augmentation: 0=none, 1=linear, 2=quadratic
)

result = pod_rbf.train(snapshot, params, config)
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `energy_threshold` | 0.99 | POD truncation threshold (0-1) |
| `truncation_method` | `"energy"` | Truncation method: `"energy"` or `"gavish_donoho"` |
| `noise_sigma` | `None` | Noise level for Gavish-Donoho (None = auto-estimate) |
| `decomposition_method` | `"auto"` | SVD algorithm: `"auto"`, `"svd"`, or `"eig"` |
| `kernel` | `"imq"` | RBF kernel type |
| `poly_degree` | 2 | Polynomial augmentation degree |
| `mem_limit_gb` | 16.0 | Memory limit for algorithm selection |
| `cond_range` | (1e11, 1e12) | Target condition number range |
| `max_bisection_iters` | 50 | Max iterations for shape optimization |

### Kernel Options

POD-RBF supports three RBF kernels:

- **`imq`** (Inverse Multi-Quadrics) - Default, good general-purpose choice
- **`gaussian`** - Smoother interpolation, requires careful shape parameter tuning
- **`polyharmonic_spline`** - No shape parameter needed, use with `kernel_order`

```python
# Using polyharmonic splines (no shape parameter optimization)
config = TrainConfig(kernel="polyharmonic_spline", kernel_order=3)
result = pod_rbf.train(snapshot, params, config)
```

## Multi-Parameter Training

For problems with multiple parameters:

```python
import numpy as np

# Parameters shape: (n_params, n_snapshots)
params = np.array([
    [100, 200, 300, 400],  # Parameter 1 (e.g., Reynolds number)
    [0.1, 0.1, 0.2, 0.2],  # Parameter 2 (e.g., Mach number)
])

result = pod_rbf.train(snapshot, params)
```

See the [2-parameter example](https://github.com/kylebeggs/POD-RBF/blob/master/examples/2-parameters.ipynb) for a complete walkthrough.

## Manual Shape Parameter

If you want to specify the RBF shape parameter instead of using automatic optimization:

```python
result = pod_rbf.train(snapshot, params, shape_factor=0.5)
```

## Understanding POD Truncation

POD extracts the dominant modes from your snapshot data. POD-RBF offers two truncation methods:

### Energy-Based Truncation (Default)

The `energy_threshold` controls how many modes are kept:

- `0.99` (default) - Keep modes until 99% of total energy is captured
- Higher values retain more modes (more accurate, slower inference)
- Lower values retain fewer modes (faster inference, may lose accuracy)

```python
config = TrainConfig(
    truncation_method="energy",  # default
    energy_threshold=0.99,
)
```

**Best for**: Clean simulation data (CFD, FEM, DNS) where noise is negligible.

### Gavish-Donoho Optimal Truncation

For noisy or experimental data, the energy-based method may retain noise modes. Gavish-Donoho uses statistical optimal hard thresholding to automatically separate signal from noise:

```python
config = TrainConfig(
    truncation_method="gavish_donoho",
)
result = pod_rbf.train(snapshot, params, config)
```

The algorithm automatically estimates the noise level from your data. If you know the noise standard deviation, you can provide it for more accurate results:

```python
config = TrainConfig(
    truncation_method="gavish_donoho",
    noise_sigma=0.1,  # Known noise standard deviation
)
```

**Best for**: Experimental measurements, PIV data, sensor data, or any snapshots with measurement noise.

**When NOT to use**: Clean simulation data—energy-based truncation is simpler and works well when noise is negligible.

### Choosing a Truncation Method

| Data Source | Recommended Method | Reason |
|-------------|-------------------|--------|
| CFD/FEM simulations | `energy` | Data is clean, energy threshold is intuitive |
| High-fidelity DNS | `energy` | Noise-free, energy captures physics |
| Experimental PIV | `gavish_donoho` | Filters measurement noise |
| Sensor measurements | `gavish_donoho` | Separates signal from sensor noise |
| Noisy simulations | `gavish_donoho` | Handles numerical noise |

### Checking Truncation Results

After training, inspect how many modes were retained:

```python
print(f"Modes retained: {result.n_modes}")
print(f"Energy retained: {result.state.truncated_energy:.4f}")
```

## Decomposition Method

POD-RBF automatically chooses between SVD and eigendecomposition based on your data shape and memory constraints. You can override this:

```python
config = TrainConfig(
    decomposition_method="auto",  # default: automatic selection
    # decomposition_method="svd",  # force SVD (faster for smaller data)
    # decomposition_method="eig",  # force eigendecomposition (memory-efficient for large data)
)
```

- **SVD**: Faster for smaller matrices, used when data fits comfortably in memory
- **Eigendecomposition**: More memory-efficient for tall matrices (many samples, few snapshots)
