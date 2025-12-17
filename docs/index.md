# POD-RBF

[![Tests](https://github.com/kylebeggs/POD-RBF/actions/workflows/tests.yml/badge.svg)](https://github.com/kylebeggs/POD-RBF/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/kylebeggs/POD-RBF/branch/master/graph/badge.svg)](https://codecov.io/gh/kylebeggs/POD-RBF)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A Python package for building Reduced Order Models (ROMs) from high-dimensional data using Proper Orthogonal Decomposition combined with Radial Basis Function interpolation.

![Lid-driven cavity results](https://raw.githubusercontent.com/kylebeggs/POD-RBF/master/examples/lid-driven-cavity/results-re-450.png)

## Features

- **JAX-based** - Enables autodifferentiation for gradient optimization, sensitivity analysis, and inverse problems
- **Shape parameter optimization** - Automatic tuning of RBF shape parameters
- **Memory-aware algorithms** - Switches between eigenvalue decomposition and SVD based on memory requirements

## Quick Install

```bash
pip install pod-rbf
```

## Quick Example

```python
import pod_rbf
import jax.numpy as jnp
import numpy as np

# Define training parameters
Re = np.array([1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

# Build snapshot matrix from CSV files
train_snapshot = pod_rbf.build_snapshot_matrix("data/train/")

# Train the model
result = pod_rbf.train(train_snapshot, Re)

# Inference on unseen parameter
sol = pod_rbf.inference_single(result.state, jnp.array(450.0))
```

## Next Steps

- [Getting Started](getting-started.md) - Installation and first steps
- [User Guide](user-guide/training.md) - Detailed usage instructions
- [API Reference](api/index.md) - Complete API documentation
- [Examples](examples.md) - Jupyter notebook examples

## References

This implementation is based on the following papers:

1. [Solving inverse heat conduction problems using trained POD-RBF network inverse method](https://www.tandfonline.com/doi/full/10.1080/17415970701198290) - Ostrowski, Bialecki, Kassab (2008)
2. [RBF-trained POD-accelerated CFD analysis of wind loads on PV systems](https://www.emerald.com/insight/content/doi/10.1108/HFF-03-2016-0083/full/html) - Huayamave et al. (2017)
3. [Real-Time Thermomechanical Modeling of PV Cell Fabrication via a POD-Trained RBF Interpolation Network](https://www.techscience.com/CMES/v122n3/38374) - Das et al. (2020)
