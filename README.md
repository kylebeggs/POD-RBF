# POD-RBF
A Python package for interpolating high-dimensional data using a Proper Orthogonal Decomposition - Radial Basis Function (POD-RBF) Network.

Given a 'snapshot' matrix of the data points with varying parameters, this code contains functions to find the truncated POD basis and interpolate using a RBF network for new parameters.

## Example

In the [example](https://github.com/kylebeggs/POD-RBF/tree/master/example) folder you can find a Python file that demonstrates how to use the package. This generates data (i.e. the snapshot matrix) and interpolates on it.

The example demonstrated here is a simple heat conduction in a 2D square domain. This was picked because an exact solution exists to the 2D heat equation with no heat generation. The exact solution is used to generate the snapshot matrix.

