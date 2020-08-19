# POD-RBF
A Python package for interpolating high-dimensional data using a Proper Orthogonal Decomposition - Radial Basis Function (POD-RBF) Network.

Given a 'snapshot' matrix of the data points with varying parameters, this code contains functions to find the truncated POD basis and interpolate using a RBF network for new parameters.

## Example

In the [example](https://github.com/kylebeggs/POD-RBF/tree/master/example) folder you can find a Python file that demonstrates how to use the package. This generates data (i.e. the snapshot matrix) and interpolates on it.

Here is a demo where we will interpolate the solution of the 2D lid-driven cavity problem. Solutions were generated using STAR-CCM+ for Reynolds numbers of 100-4000 in increments of 500. The Re number will serve as our single parameter in this case.
