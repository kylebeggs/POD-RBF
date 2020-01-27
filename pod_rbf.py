import numpy as np
import time
from tqdm import tqdm
import gc


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
            truncId = i + 1
            break
    if truncId is not None:
        eigVals = eigVals[:truncId]
        energy = energy[:truncId]
    eigVecs = eigVecs[:, :truncId]

    # calculate the truncated POD basis
    basis = np.matmul(snapShot, eigVecs)
    for i in range(len(eigVals)):
        basis[:, i] = basis[:, i] / np.sqrt(eigVals[i])

    print('took {:3.3f} sec'.format(time.time() - start))

    gc.collect()

    return basis


def mkHardyIMQTrainMatrix(trainParams, shapeFactor):
    """Make the Radial Basis Function (RBF) matrix using the
     Hardy Inverse Multi-Qualdrics (IMQ) function

    Parameters
    ----------
    trainParams : ndarray
        The parameters used to generate the snapshot matrix.
    shapeFactor : float
        The shape factor to be used in the RBF network.

    Returns
    -------
    ndarray
        The RBF matrix.

    """
    print('constructing the RBF matrix... ', end='')
    start = time.time()

    sum = np.zeros((len(trainParams[0, :]), len(trainParams[0, :])))
    for i in tqdm(range(trainParams.shape[0])):
        I, J = np.meshgrid(trainParams[i, :],
                           trainParams[i, :],
                           indexing='ij',
                           copy=False)
        sum += np.abs(I - J)
    gc.collect()
    print('took {} sec'.format(time.time() - start))

    return 1 / np.sqrt(sum + shapeFactor**2)


def mkHardyIMQInfMatrix(trainParams, infParams, shapeFactor):
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
    print('constructing the RBF matrix... ', end='')
    start = time.time()

    sum = np.zeros((1, len(trainParams[0, :])))
    for i in tqdm(range(trainParams.shape[0])):
        I, J = np.meshgrid(infParams[i],
                           trainParams[i, :],
                           indexing='ij',
                           copy=False)
        sum += np.abs(I - J)
    gc.collect()
    return 1 / np.sqrt(sum + shapeFactor**2)

    #I1, J1 = np.meshgrid(infParams[0], trainParams[0, :], indexing='ij')
    #I2, J2 = np.meshgrid(infParams[1], trainParams[1, :], indexing='ij')
    #I3, J3 = np.meshgrid(infParams[2], trainParams[2, :], indexing='ij')
    #sum = np.abs(I1 - J1) + np.abs(I2 - J2) + np.abs(I3 - J3)
    #del I1, J1, I2, J2, I3, J3
    #sum = sum.reshape(len(trainParams[0, :]), )
    #gc.collect()
    #return 1 / np.sqrt(sum + shapeFactor**2)


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

    # build the Radial Basis Function (RBF) matrix
    F = mkHardyIMQTrainMatrix(trainParams, shapeFactor)
    print('Conditioning Number = {} ... '.format(np.linalg.cond(F) / 1e9),
          end='')

    # calculate the amplitudes (A) and weights/coefficients (B)
    A = np.matmul(np.transpose(basis), snapShot)
    try:
        B = np.matmul(A, np.linalg.pinv(np.transpose(F)))
    except:
        print('failed!!!!!!')
        quit()

    print('took {:3.3f} sec'.format(time.time() - start))

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
    F = mkHardyIMQInfMatrix(trainParams, infParams, shapeFactor)

    # calculate the inferenced solution
    A = np.matmul(weights, np.transpose(F))
    inference = np.matmul(basis, A)

    print('took {:3.3f} sec'.format(time.time() - start))

    return inference.astype(np.uint8)
