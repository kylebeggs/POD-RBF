import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt


def buildSnapshotMatrix(params, num_points):
    """
    Assemble the snapshot matrix
    """
    print("making the snapshot matrix... ", end="")
    start = time.time()

    # evaluate the analytical solution
    num_terms = 50
    L = 1
    T_L = params[0, :]

    n = np.arange(0, num_terms)
    # calculate lambdas
    lambs = np.pi * (2 * n + 1) / (2 * L)

    # define points
    x = np.linspace(0, L, num=num_points)
    X, Y = np.meshgrid(x, x, indexing="xy")

    snapshot = np.zeros((num_points ** 2, len(T_L)))
    for i in range(len(T_L)):
        # calculate constants
        C = (
            8
            * T_L[i]
            * (2 * (-1) ** n / (lambs * L) - 1)
            / ((lambs * L) ** 2 * np.cosh(lambs * L))
        )
        T = np.zeros_like(X)
        for j in range(0, num_terms):
            T = T + C[j] * np.cosh(lambs[j] * X) * np.cos(lambs[j] * Y)
        snapshot[:, i] = T.flatten()

    print("took {:3.3f} sec".format(time.time() - start))

    return snapshot


if __name__ == "__main__":

    import jax
    import jax.numpy as jnp
    import pod_rbf

    jax.config.update('jax_default_device', jax.devices('cpu')[0])  # Change to 'gpu' or 'tpu' for accelerators

    T_L = np.linspace(1, 100, num=11)
    T_L = np.expand_dims(T_L, axis=0)
    T_L_test = 55.0
    num_points = 41

    # make snapshot matrix
    snapshot = buildSnapshotMatrix(T_L, num_points)

    # calculate 'test' solution
    # evaluate the analytical solution
    num_terms = 50
    L = 1
    n = np.arange(0, num_terms)
    # calculate lambdas
    lambs = np.pi * (2 * n + 1) / (2 * L)
    # define points
    x = np.linspace(0, L, num=num_points)
    X, Y = np.meshgrid(x, x, indexing="xy")
    # calculate constants
    C = (
        8
        * T_L_test
        * (2 * (-1) ** n / (lambs * L) - 1)
        / ((lambs * L) ** 2 * np.cosh(lambs * L))
    )
    T_test = np.zeros_like(X)
    for n in range(0, num_terms):
        T_test = T_test + C[n] * np.cosh(lambs[n] * X) * np.cos(lambs[n] * Y)

    # train the POD-RBF model
    config = pod_rbf.TrainConfig(energy_threshold=0.5, poly_degree=2)
    result = pod_rbf.train(snapshot, T_L, config)
    state = result.state

    # inference the trained model
    sol = pod_rbf.inference_single(state, jnp.array(T_L_test))

    print("Energy kept after truncating = {}%".format(state.truncated_energy))
    print("Cumulative Energy = {}%".format(state.cumul_energy))

    fig = plt.figure(figsize=(12, 9))
    c = plt.pcolormesh(T_test, cmap="magma")
    fig.colorbar(c)

    fig = plt.figure(figsize=(12, 9))
    c = plt.pcolormesh(sol.reshape((num_points, num_points)), cmap="magma")
    fig.colorbar(c)

    fig = plt.figure(figsize=(12, 9))
    diff = np.abs(sol.reshape((num_points, num_points)) - T_test) / T_test * 100
    c = plt.pcolormesh(diff, cmap="magma")
    fig.colorbar(c)

    plt.show()
