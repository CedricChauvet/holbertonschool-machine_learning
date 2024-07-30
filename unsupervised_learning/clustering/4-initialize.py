#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    intialze my gaussian
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    n, d = X.shape

    if not isinstance(k, int) or k <= 0:
        return None

    m, _ = kmeans(X, k, iterations=1000)

    pi = np.zeros(k)
    pi[:] = np.round(1 / k, decimals=8)
    s = np.repeat([np.identity(d)], repeats=k, axis=0)

    return pi, m, s
