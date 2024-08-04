#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    analyze variance wrt numbers of clusters
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    n, d = X.shape

    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if not isinstance(kmax, int):
        return None, None

    results = []
    d_var = []

    if kmax is None or kmax >= n:
        kmax = n

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        var = np.sum((X[:] - C[clss]) ** 2)

        if k == kmin:
            var_min = var
        results.append((C, clss))

        d_var.append(var_min - var)

    return results, d_var
