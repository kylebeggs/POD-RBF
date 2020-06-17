import sys
import time
import numpy as np
import matplotlib.pyplot as plt


def mkSnapshotMatrix(params):
    """
    Assemble the snapshot matrix
    """
    print("making the snapshot matrix... ", end="")
    start = time.time()

    # evaluate the analytical solution
    num_terms = 50
    num_points = 201
    L = 1
    T_L = params[0, :]

    n = np.arange(0, num_terms)
    # calculate lambdas
    lambs = np.pi * (2 * n + 1) / (2 * L)

    # define points
    x = np.linspace(0, L, num=num_points)
    X, Y = np.meshgrid(x, x, indexing="xy")

    snapShot = np.zeros((num_points ** 2, len(T_L)))
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
        snapShot[:, i] = T.flatten()
        del C
    

    print("took {:3.3f} sec".format(time.time() - start))

    return snapShot


if __name__ == "__main__":

    sys.path.insert(0, "/media/kylebeggs/samsungt5/pro/code/pod")
    import pod_rbf as p

    T_L = np.linspace(1, 100, num=10)
    T_L = np.expand_dims(T_L, axis=0)

    shapeFactor = 900
    energyThreshold = 0.99

    # make snapshot matrix
    snapShot = mkSnapshotMatrix(T_L)

    # calculate truncated POD basis
    basis = p.calcTruncatedPODBasis(snapShot, energyThreshold)

    # calculate 'test' solution
    # evaluate the analytical solution
    num_terms = 50
    num_points = 201
    L = 1
    T_L_test = np.array([55])
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

    # inference the trained RBF network
    weights = p.trainRBF(snapShot, basis, shapeFactor, T_L)
    sol = p.infRBF(basis, weights, shapeFactor, T_L, T_L_test)

    fig = plt.figure(figsize=(12, 9))
    c = plt.pcolormesh(T_test, cmap="magma")
    fig.colorbar(c)

    fig = plt.figure(figsize=(12, 9))
    c = plt.pcolormesh(sol.reshape((num_points, num_points)), cmap="magma")
    fig.colorbar(c)

    fig = plt.figure(figsize=(12, 9))
    diff = (sol.reshape((num_points, num_points))-T_test)
    c = plt.pcolormesh(diff, cmap="magma")
    fig.colorbar(c)

    plt.show()

    print("error ={}".format(np.linalg.norm(sol - T_test.flatten(), np.inf)))

