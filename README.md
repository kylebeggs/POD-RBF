# POD-RBF

[![Tests](https://github.com/kylebeggs/POD-RBF/actions/workflows/tests.yml/badge.svg)](https://github.com/kylebeggs/POD-RBF/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/kylebeggs/POD-RBF/branch/master/graph/badge.svg)](https://codecov.io/gh/kylebeggs/POD-RBF)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

![Re-450](examples/lid-driven-cavity/results-re-450.png)

A Python package for building a Reduced Order Model (ROM) from high-dimensional data using a Proper
Orthogonal Decomposition - Radial Basis Function (POD-RBF) Network.

Given a 'snapshot' matrix of the data points with varying parameters, this code contains functions
to find the truncated POD basis and interpolate using a RBF network for new parameters.

This code is based on the following papers implementing the method:

1. [Solving inverse heat conduction problems using trained POD-RBF network inverse method - Ostrowski, Bialecki, Kassab (2008)](https://www.tandfonline.com/doi/full/10.1080/17415970701198290)
2. [RBF-trained POD-accelerated CFD analysis of wind loads on PV systems - Huayamave, Ceballos, Barriento, Seigneur, Barkaszi, Divo, and Kassab (2017)](https://www.emerald.com/insight/content/doi/10.1108/HFF-03-2016-0083/full/html)
3. [Real-Time Thermomechanical Modeling of PV Cell Fabrication via a POD-Trained RBF Interpolation Network - Das, Khoury, Divo, Huayamave, Ceballos, Eaglin, Kassab, Payne, Yelundur, and Seigneur (2020)](https://www.techscience.com/CMES/v122n3/38374)

Features:

* **JAX-based** - enables autodifferentiation for gradient optimization, sensitivity analysis, and inverse problems
* Shape parameter optimization for the Radial Basis Functions (RBFs)
* Algorithm switching based on memory requirements (eigenvalue decomposition vs. SVD)

## Installation

```bash
pip install pod-rbf
# or
uv add pod-rbf
```

## Example

In the [example](https://github.com/kylebeggs/POD-RBF/tree/master/examples) folder you can find two
examples of that demonstrates how to use the package. The first is a simple heat conduction problem
on a unit square.

The other example will be demonstrated step-by-step here. We seek to build a ROM of the 2D
lid-driven cavity problem. For the impatient, here is the full code to run this example. I will
break down each line in the sections below.

If you wish to build a ROM with multiple parameters, see this basic [2-parameter example](https://github.com/kylebeggs/POD-RBF/tree/master/examples/2-parameters.ipynb).

```python
import pod_rbf
import jax.numpy as jnp
import numpy as np

Re = np.linspace(0, 1000, num=11)
Re[0] = 1

# make snapshot matrix from csv files
train_snapshot = pod_rbf.build_snapshot_matrix("examples/lid-driven-cavity/data/train/")

# train the model (keeps 99% energy in POD modes by default)
result = pod_rbf.train(train_snapshot, Re)

# inference on an unseen parameter
sol = pod_rbf.inference_single(result.state, jnp.array(450.0))
```

### Building the snapshot matrix

First, we need to build the snapshot matrix, X, which contains the data we are training on. It must be of the form where each column is the k-th 'snapshot' of the solution field given some
parameter, p_k, with n samples in the snapshot at locations x_n. A single snapshot is below

![snapshot equation](examples/lid-driven-cavity/eq-snapshot.png)

and the snapshot matrix would then look like

![snapshot equation](examples/lid-driven-cavity/eq-snapshot-matrix.png)

where m is the total number of snapshots.

For example, suppose our lid-driven cavity was solved on a mesh with 400 cells and we varied the
parameter of interest (Re number in this case) 10 times. We would have a matrix of size (n,m) =
(400,10).

For our example, solutions were generated using STAR-CCM+ for Reynolds numbers of 1-1000 in
increments of 100. The Re number will serve as our single parameter in this case. Snapshots were
generated as a separate .csv file for each. To make it easier to combine them all into the snapshot
matrix, there is a function which takes the path and file pattern. The same syntax is borrowed from
the ffmpeg tool - that is, if you had files named as sample_001.csv, sample_002.csv ... you would
input sample_%03d.csv. The files for this example are named as re-%04d.csv so we would issue a
command as

```python
>>> import pod_rbf
>>> train_snapshot = pod_rbf.build_snapshot_matrix("examples/lid-driven-cavity/data/train/")
```

---
Note: if you are using this approach where each snapshot is contained in a different csv file,
please group all of them into a directory of their own.

---

If you notice, these files are contained in the train folder, as I also generated some more
snapshots for validation (which as you probably guessed is in the /data/validation folder). now we
need to generate the array of input parameters that correspond to each snapshot.

```python
>>> Re = np.linspace(0, 1000, num=11)
>>> Re[0] = 1
```

---
Note: it is extremely important that each input parameter maps to the same element number of the
snapshot matrix. For example if the 5th column (index 4) then the input parameter used to generate
that snapshot should be what you find in the 5th element (index 4) of the array, e.g.
```train_snapshot[:,4] -> Re[4]```. The csv files are loaded in alpha-numeric order so that is why
the input parameter array goes from  1 -> 1000.

---

where ```Re``` is an array of input parameters that we are training the model on. Next, we train
the model with a single function call. We choose to keep 99% of the energy in POD modes (this is
the default, so you don't have to set that).

```python
>>> result = pod_rbf.train(train_snapshot, Re)
>>> # Or with custom config:
>>> config = pod_rbf.TrainConfig(energy_threshold=0.99)
>>> result = pod_rbf.train(train_snapshot, Re, config)
```

Now that the weights and truncated POD basis have been calculated and stored in `result.state`, we
can inference on the model using any input parameter.

```python
>>> import jax.numpy as jnp
>>> sol = pod_rbf.inference_single(result.state, jnp.array(450.0))
```

and we can plot the results comparing the inference and target below

![Re-450](examples/lid-driven-cavity/results-re-450.png)

and for Reynold's number of 50:

![Re-450](examples/lid-driven-cavity/results-re-50.png)


### Saving and loading models
You can save and load the trained model state:

```python
>>> pod_rbf.save_model("model.pkl", result.state)
>>> state = pod_rbf.load_model("model.pkl")
>>> sol = pod_rbf.inference_single(state, jnp.array(450.0))
```

### Autodifferentiation

Since POD-RBF is built on JAX, you can compute gradients for optimization and inverse problems:

```python
>>> import jax
>>> grad_fn = jax.grad(lambda p: jnp.sum(pod_rbf.inference_single(result.state, p)**2))
>>> gradient = grad_fn(jnp.array(450.0))
```
