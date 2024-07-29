#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np


def initialize(X, k):
    """
    set the centroids
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be an array")
    if not isinstance(k, int) or k <= 0:
        raise ValueError(" K must be a positive int")

    n, d = X.shape

    init = np.random.uniform(low=np.min(X, axis=0),
                             high=np.max(X, axis=0), size=(k, d))
    return init
