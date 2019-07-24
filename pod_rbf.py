import numpy as np
import time



def calcTruncatedPODBasis(snapShot, energyThreshold):
    """Calculate the truncated POD basis.

    :param ndarray snapShot: the matrix containing data points for each parameter as columns
    :param float energyThreshold: the percent of energy to keep in the system
    :returns: the truncated POD basis
    :rtype: ndarray
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


def trainRBF(snapShot, basis, shapeFactor, trainParams):
    """
    Train the Radial Basis Function (RBF) network

    :param ndarray snapShot: the matrix containing data points for each parameter as columns
    :param ndarray basis: the truncated POD basis
    :param float shapeFactor: the shape factor to be used in the RBF network
    :param ndarray trainParams: the parameters used to generate the snapshot matrix
    :returns: the weights/coefficients of the RBF network
    :rtype: ndarray
    """
    print('training the RBF network... ', end='')
    start = time.time()

    c = shapeFactor
    numParams = len(trainParams)

    # build the Radial Basis Function (RBF) matrix
    F = np.zeros( (numParams,numParams) )
    for i in range(numParams):
        for j in range(numParams):
            F[i,j] = 1/np.sqrt( np.linalg.norm(trainParams[i]-trainParams[j]) + c**2 )

    # calculate the amplitudes (A) and weights/coefficients (B)
    A = np.matmul(np.transpose(basis), snapShot)
    B = np.matmul(A,np.linalg.inv(F))

    print('took {:3.3f} sec'.format(time.time()-start))

    return B


def infRBF(basis, weights, shapeFactor, trainParams, infParams):
    """Inference the RBF network with an unseen parameter.

    :param ndarray basis: the truncated POD basis
    :param ndarray weights: the weights/coefficients of the RBF network
    :param float shapeFactor: the shape factor to be used in the RBF network
    :param ndarray trainParams: the parameters used to generate the snapshot matrix
    :param ndarray infParams: the parameters to inference the RBF network on
    :returns: the output of the RBF netowkr according to the infParams argument
    :rtype: ndarray
    """
    print('inferencing the RBF network... ', end='')
    start = time.time()

    c = shapeFactor

    if hasattr(infParams, '__len__'):
        numInfParams = len(infParams)
    else:
        numInfParams = 1
    numTrainParams = len(trainParams)

    # build the Radial Basis Function (RBF) matrix
    if numInfParams > 1:
        F = np.zeros( (numInfParams,numTrainParams) )
        for i in range(numInfParams):
            for j in range(numTrainParams):
                F[i,j] = 1/np.sqrt( np.linalg.norm(infParams[i]-trainParams[j]) + c**2 )
    else:
        F = np.zeros(numTrainParams)
        for j in range(numTrainParams):
            F[j] = 1/np.sqrt( np.linalg.norm(infParams-trainParams[j]) + c**2 )


    # calculate the inferenced solution
    A = np.matmul(weights, np.transpose(F))
    inference = np.matmul(basis, A)

    print('took {:3.3f} sec'.format(time.time()-start))

    return inference



