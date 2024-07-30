#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np


def variance(X, C):
    """
    calculate variance
    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the
    centroid means for each cluster
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    n, d1 = X.shape

    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    k, d2 = C.shape

    if not isinstance(k, int) or k <= 0:
        return None

    if d1 != d2:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    var = np.sum((X[:] - C[clss]) ** 2)

    return var
