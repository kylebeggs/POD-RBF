import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import gc
import psutil


def loadDOE(path):
    designNum, Exit, ExitPress, RefPress = np.loadtxt(path,
                                                      delimiter=',',
                                                      usecols=(0, 3, 5, 6),
                                                      skiprows=1,
                                                      unpack=True)
    return np.stack([designNum, Exit, ExitPress, RefPress], axis=0)


def printMem():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1e9
    print('mem used: {:3.2f} GB'.format(mem))


def mkSnapshotAndParams(caseDir, doe, percent=0.1):
    """
    Assemble the snapshot matrix
    """
    print('making the snapshot matrix... ', end='')
    start = time.time()

    dirList = os.listdir(caseDir)
    random.shuffle(dirList)
    cases = dirList[:int(percent * len(dirList))]

    snapShot = np.zeros((512 * 256 * 3, len(cases)))
    trainParams = np.zeros((512 * 256 * 3, len(cases)))
    #trainParams = np.zeros((3, len(cases)))
    for i in range(len(cases)):

        id = ''
        temp = cases[i]
        for c in temp:
            if c.isdigit():
                id += str(c)
        id = int(id)

        img = np.load(os.path.join(caseDir, 'case_' + str(id) + '.npz'),
                      mmap_mode='r')
        img = img['a']

        inputs = img[:3, :, :].astype(np.uint8)
        targets = img[3:, :, :].astype(np.uint8)

        snapShot[:, i] = np.transpose(
            np.hstack((targets[0, :, :].flatten(), targets[1, :, :].flatten(),
                       targets[2, :, :].flatten())))
        trainParams[:, i] = np.transpose(
            np.hstack((inputs[0, :, :].flatten(), inputs[1, :, :].flatten(),
                       inputs[2, :, :].flatten())))

        gc.collect()

        #trainParams[:, i] = np.array(
        #    [doe[1, id - 1], doe[2, id - 1], doe[3, id - 1]])

    print('took {:3.3f} sec'.format(time.time() - start))

    return snapShot, trainParams


if __name__ == "__main__":

    sys.path.insert(0, os.getcwd())
    import pod_rbf as p

    # TODO - user settings
    dataDir = 'train/'
    paramsPath = 'params.csv'
    energyThreshold = 0.99
    shapeFactor = 1e4
    doe = loadDOE(paramsPath)

    # make snapshot matrix
    #snapShot, trainParams = mkSnapshotAndParams(dataDir, doe, percent=0.3)
    #np.savez('snapShot.npz', a=snapShot)
    #np.savez('trainParams.npz', a=trainParams)
    printMem()
    temp = np.load('snapShot.npz', mmap_mode='r')
    snapShot = temp['a']
    temp = np.load('trainParams.npz', mmap_mode='r')
    trainParams = temp['a']
    del temp

    # calculate truncated POD basis
    basis = p.calcTruncatedPODBasis(snapShot, energyThreshold)

    # train the RBF network
    #weights = p.trainRBF(snapShot, basis, shapeFactor, trainParams)
    #np.savez('weights.npz', a=weights)
    printMem()

    temp = np.load('weights.npz', mmap_mode='r')
    weights = temp['a']
    del temp

    # calculate 'test' solution
    id = 2500
    try:
        img = np.load('train/case_' + str(id) + '.npz', mmap_mode='r')
    except:
        img = np.load('test/case_' + str(id) + '.npz', mmap_mode='r')
    img = img['a']
    inputs = img[:3, :, :].astype(np.uint8)
    targets = img[3:, :, :].astype(np.uint8)
    #testInputs = np.array([doe[1, id - 1], doe[2, id - 1], doe[3, id - 1]])
    testTarget = np.transpose(
        np.hstack((targets[0, :, :].flatten(), targets[1, :, :].flatten(),
                   targets[2, :, :].flatten())))
    testInputs = np.transpose(
        np.hstack((inputs[0, :, :].flatten(), inputs[1, :, :].flatten(),
                   inputs[2, :, :].flatten())))

    # inference the trained RBF network
    sol = p.infRBF(basis, weights, shapeFactor, trainParams, testInputs)
    sol = sol.reshape(sol.shape[0], )
    printMem()

    out = sol.reshape(3, 256, 512)
    for i in range(3):
        out[i, :, :] = np.where(inputs[2, :, :] == 0, 0, out[i, :, :])

    error = np.abs(out[:, :, :] - targets[:, :, :])
    print('error min = {}'.format(np.min(error)))
    print('error max = {}'.format(np.max(error)))

    for i in range(3):
        print(out[i, 125, 50])
        print(targets[i, 125, 50])
        print('\n')

    for i in range(3):
        fig, ax = plt.subplots(3, 1, figsize=(15, 25))
        vmin = 0
        vmax = 255
        pos1 = ax[0].imshow(targets[i, :, :], vmin=vmin, vmax=vmax)
        pos2 = ax[1].imshow(out[i, :, :], vmin=vmin, vmax=vmax)
        pos3 = ax[2].imshow(error[i, :, :], vmin=vmin, vmax=vmax)
        mask = np.where(inputs[2, :, :] > 0, 1, 0)
        print('error[{}] = {}'.format(
            i,
            np.mean(error[i, :, :]) / np.mean(targets[i, :, :])))
        fig.colorbar(pos1, ax=ax[0])
        fig.colorbar(pos2, ax=ax[1])
        fig.colorbar(pos3, ax=ax[2])

    plt.show()
