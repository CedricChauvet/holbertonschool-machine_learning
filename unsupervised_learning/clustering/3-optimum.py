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

    n, d = X.shape
    results = []
    d_var = []

    if kmax is None or kmax >= n:
        kmax = n

    for k in range(kmin, kmax+ 1 ):
        C, clss = kmeans(X, k, iterations)
        var = variance(X, C)

        if k == kmin:
            var_min = var
        results.append(C)
        results.append(clss)
        d_var.append(var_min - var)

    return results, d_var
