import numpy as np
import time



def calcTruncatedPODBasis(snapShot, energyThreshold):
    """Calculate the truncated POD basis.

    Parameters
    ----------
    snapShot : ndarray
        The matrix containing data points for each parameter as columns.
    energyThreshold : float
        The percent of energy to keep in the system.

    Returns
    -------
    ndarray
        The truncated POD basis.

    """
    print('calculating the truncated POD basis... ', end='')
    start = time.time()

    # compute the covarience matrix and corresponding eigenvalues and eigenvectors
    cov = np.matmul(np.transpose(snapShot), snapShot)
    eigVals, eigVecs = np.linalg.eigh(cov)
    eigVals = np.abs(eigVals.real)
    eigVecs = eigVecs.real

    # compute the energy in the system
    energy = eigVals / np.sum(eigVals)

    # truncate the eigenvalues and eigenvectors
    totalEnergy = 0
    for i in range(len(energy)):
        totalEnergy += energy[i]
        if totalEnergy > energyThreshold:
            truncId = i+1
            break
    if truncId is not None:
        eigVals = eigVals[:truncId]
        energy = energy[:truncId]
    eigVecs = eigVecs[:,:truncId]

    # calculate the truncated POD basis
    basis = np.matmul(snapShot, eigVecs)
    for i in range(len(eigVals)):
        basis[:,i] = basis[:,i]/np.sqrt(eigVals[i])

    print('took {:3.3f} sec'.format(time.time()-start))

    return basis


def mkHardyIMQMatrix(trainParams, infParams, shapeFactor):
    """Make the Radial Basis Function (RBF) matrix using the
     Hardy Inverse Multi-Qualdrics (IMQ) function

    Parameters
    ----------
    trainParams : ndarray
        The parameters used to generate the snapshot matrix.
    infParams : ndarray
        The parameters to inference the RBF network on.
    shapeFactor : float
        The shape factor to be used in the RBF network.

    Returns
    -------
    ndarray
        The RBF matrix.

    """
    if hasattr(infParams, '__len__'):
        I, J = np.meshgrid(infParams, trainParams, indexing='ij')
        return 1/np.sqrt( np.abs(I-J) + shapeFactor**2 )
    else:
        return 1/np.sqrt( np.abs(infParams-trainParams) + shapeFactor**2 )


def trainRBF(snapShot, basis, shapeFactor, trainParams):
    """
    Train the Radial Basis Function (RBF) network

    Parameters
    ----------
    snapShot : ndarray
        The matrix containing data points for each parameter as columns.
    basis : ndarray
        The truncated POD basis.
    shapeFactor : float
        The shape factor to be used in the RBF network.
    trainParams : ndarray
        The parameters used to generate the snapshot matrix.

    Returns
    -------
    ndarray
        The weights/coefficients of the RBF network.

    """
    print('training the RBF network... ', end='')
    start = time.time()

    n = len(trainParams)

    # build the Radial Basis Function (RBF) matrix
    F = mkHardyIMQMatrix(trainParams, trainParams, shapeFactor)

    # calculate the amplitudes (A) and weights/coefficients (B)
    A = np.matmul(np.transpose(basis), snapShot)
    B = np.matmul(A,np.linalg.inv(F))

    print('took {:3.3f} sec'.format(time.time()-start))

    return B


def infRBF(basis, weights, shapeFactor, trainParams, infParams):
    """Inference the RBF network with an unseen parameter.

    Parameters
    ----------
    basis : ndarray
        The truncated POD basis.
    weights : ndarray
        The weights/coefficients of the RBF network.
    shapeFactor : float
        The shape factor to be used in the RBF network.
    trainParams : ndarray
        The parameters used to generate the snapshot matrix.
    infParams : ndarray
        The parameters to inference the RBF network on.

    Returns
    -------
    ndarray
        The output of the RBF netowkr according to the infParams argument.

    """
    print('inferencing the RBF network... ', end='')
    start = time.time()

    # build the Radial Basis Function (RBF) matrix
    F = mkHardyIMQMatrix(trainParams, infParams, shapeFactor)

    # calculate the inferenced solution
    A = np.matmul(weights, np.transpose(F))
    inference = np.matmul(basis, A)

    print('took {:3.3f} sec'.format(time.time()-start))

    return inference



