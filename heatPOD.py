import numpy as np
import sys, os
import matplotlib.pyplot as plt
import time



def mkSnapshotMatrix(numNodes, L, T0):
    """
    Assemble the snapshot matrix
    """
    print('making the snapshot matrix... ', end='')
    start = time.time()

    snapShot = np.zeros( (numNodes, len(T0)) )
    for i in range(len(T0)):
        prob = heat.problem(numNodes, L/(numNodes-1), T0[i])
        A, d = heat.mkFiniteVolumeMatrices(prob)
        snapShot[:,i] = np.linalg.solve(A, d)

    print('took {:3.3f} sec'.format(time.time()-start))
    
    return snapShot


def calcTruncatedPODBasis(snapShot):
    """
    Calculate the truncated POD basis
    """
    print('calculating the truncated POD basis... ', end='')
    start = time.time()

    cov = np.dot(np.transpose(snapShot), snapShot)
    eigVals, eigVecs = np.linalg.eigh(cov)
    eigVals = np.abs(eigVals.real)
    eigVecs = eigVecs.real

    # compute the energy in the system
    energy = eigVals / np.sum(eigVals)

    # truncate the eigenvalues and eigenvectors
    threshold = 0.999
    totalEnergy = 0
    for i in range(len(energy)):
        totalEnergy += energy[i]
        if totalEnergy > threshold:
            truncId = i+1
            break
    if truncId is not None:
        eigVals = eigVals[:truncId]
        energy = energy[:truncId]

    # compute eigenvecs
    eigVecs = eigVecs[:,:truncId]

    basis = np.matmul(snapShot, eigVecs)
    for i in range(len(eigVals)):
        basis[:,i] = basis[:,i]/np.sqrt(eigVals[i])

    print('took {:3.3f} sec'.format(time.time()-start))

    return basis


def trainRBF(snapShot, basis, shapeFactor, trainParams):
    """
    Train the Radial Basis Function (RBF) network
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
    """
    Inference the RBF network with an unseen parameter.
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
    sol = np.matmul(basis, A)

    print('took {:3.3f} sec'.format(time.time()-start))

    return sol



if __name__ == "__main__":

    sys.path.insert(0, '/home/kylebeggs/CERT/kylebeggs/phd/mahi/meshless')
    import heatConduction1D as heat

    L = 1
    numNodes = 10

    T0 = np.linspace(20, 1000, num=500)

    shapeFactor = 10

    # make snapshot matrix
    snapShot = mkSnapshotMatrix(numNodes, L, T0)

    # calculate truncated POD basis
    basis = calcTruncatedPODBasis(snapShot)
    
    # calculate 'test' solution
    testT0 = 85
    prob = heat.problem(numNodes, L/(numNodes-1), testT0)
    A, d = heat.mkFiniteVolumeMatrices(prob)
    test = np.linalg.solve(A, d)
    
    # inference the trained RBF network
    weights = trainRBF(snapShot, basis, shapeFactor, T0)
    sol = infRBF(basis, weights, shapeFactor, T0, testT0)

    print('\n')
    print(test)
    print(sol)
    print( np.divide(np.abs(test-sol), ((test+sol)/2) )*100)



