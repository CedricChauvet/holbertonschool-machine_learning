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
    m, clss = kmeans(X, k, iterations=1000)
    pi = [1 / k] * k
    s = np.repeat([np.identity(2)], repeats=k, axis=0)

    return pi, m, s
