# Examples

## Jupyter Notebooks

Explore these example notebooks to see POD-RBF in action:

### Lid-Driven Cavity

A complete walkthrough using CFD data from a 2D lid-driven cavity simulation at various Reynolds numbers.

[:octicons-mark-github-16: View on GitHub](https://github.com/kylebeggs/POD-RBF/tree/master/examples/lid-driven-cavity){ .md-button }

**What you'll learn:**

- Building a snapshot matrix from CSV files
- Training a single-parameter model
- Visualizing predictions vs. ground truth

### Multi-Parameter Example

Training a model with two input parameters.

[:octicons-mark-github-16: View on GitHub](https://github.com/kylebeggs/POD-RBF/blob/master/examples/2-parameters.ipynb){ .md-button }

**What you'll learn:**

- Setting up multi-parameter training data
- Inference with multiple parameters
- Parameter space exploration

### Heat Conduction

A simple heat conduction problem on a unit square.

[:octicons-mark-github-16: View on GitHub](https://github.com/kylebeggs/POD-RBF/tree/master/examples/heat-conduction){ .md-button }

**What you'll learn:**

- Basic POD-RBF workflow
- Working with thermal simulation data

### Shape Parameter Optimization

Exploring RBF shape parameter selection.

[:octicons-mark-github-16: View on GitHub](https://github.com/kylebeggs/POD-RBF/tree/master/examples/shape-optimization){ .md-button }

**What you'll learn:**

- How shape parameters affect interpolation
- Automatic vs. manual shape parameter selection

## Running the Examples

Clone the repository and install the package:

```bash
git clone https://github.com/kylebeggs/POD-RBF.git
cd POD-RBF
pip install -e .
```

Then open the Jupyter notebooks in the `examples/` directory:

```bash
jupyter notebook examples/
```
