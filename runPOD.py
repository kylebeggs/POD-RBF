import sys
import time
import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":

    sys.path.insert(0, '/home/kylebeggs/CERT/kylebeggs/mahi/meshless')
    sys.path.insert(0, '/home/kylebeggs/CERT/kylebeggs/mahi/pod')
    import heatConduction1D as heat
    import pod_rbf as p
    
    L = 1
    numNodes = 100
    
    T0 = np.linspace(20, 1000, num=50)
    
    shapeFactor = 10
    energyThreshold = 0.999
    
    # make snapshot matrix
    snapShot = mkSnapshotMatrix(numNodes, L, T0)
    
    # calculate truncated POD basis
    basis = p.calcTruncatedPODBasis(snapShot, energyThreshold)
    
    # calculate 'test' solution
    testT0 = 85
    prob = heat.problem(numNodes, L/(numNodes-1), testT0)
    A, d = heat.mkFiniteVolumeMatrices(prob)
    test = np.linalg.solve(A, d)
    
    # inference the trained RBF network
    weights = p.trainRBF(snapShot, basis, shapeFactor, T0)
    sol = p.infRBF(basis, weights, shapeFactor, T0, testT0)
    
    error = np.divide(np.abs(test-sol), ((test+sol)/2), out=np.zeros_like(test), where=test!=0)*100
    print('Avg error = {:3.5f}'.format(np.mean(error)))

    plt.plot(test)
    plt.plot(sol)
    plt.show()
    



