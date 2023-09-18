import numpy as np
from multiprocessing.pool import Pool


def _gradFiniteDifference1(centerPoint, function, delta, nbProcesses):
    allPoints = [centerPoint]
    allVariations = [0.]

    nbParam = centerPoint.shape[0]

    for j, val in enumerate(centerPoint):
        dpar = max(1.e-8, abs(delta*val))
        allVariations.append(dpar)

        local = centerPoint.copy()
        local[j] += dpar
        allPoints.append(local)

    allResults = []
    with Pool(nbProcesses) as pool:
        allResults = pool.map(function, allPoints)

    jac = np.zeros((allResults[0].shape[0], nbParam))

    for j, (col, dpar) in enumerate(zip(allResults[1:], allVariations[1:])):
        jac[:, j] = (col - allResults[0])/dpar

    return jac


def _gradFiniteDifference2(centerPoint, function, delta, nbProcesses):
    allPoints = []
    allVariations = []

    nbParam = centerPoint.shape[0]

    for j, val in enumerate(centerPoint):
        dpar = max(1.e-8, abs(delta*val))
        allVariations.append(2.*dpar)

        localplus = centerPoint.copy()
        localplus[j] += dpar
        allPoints.append(localplus)

        localminus = centerPoint.copy()
        localminus[j] -= dpar
        allPoints.append(localminus)

    allResults = []
    with Pool(nbProcesses) as pool:
        allResults = pool.map(function, allPoints)

    jac = np.zeros((allResults[0].shape[0], nbParam))
    for j, dpar in enumerate(allVariations):
        jac[:, j] = (allResults[2*j]-allResults[2*j+1])/dpar

    return jac


def grad_function(p, f, nbProcess=1, method="fd1"):

    delta = np.sqrt(np.finfo(np.float64).eps)

    if method == "fd1":
        return _gradFiniteDifference1(p, f, delta, nbProcess)
    elif method == "fd2":
        return _gradFiniteDifference1(p, f, delta, nbProcess)
    else:
        raise NotImplementedError
