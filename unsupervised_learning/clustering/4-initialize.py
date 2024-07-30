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
    m, _ = kmeans(X, k, iterations=1000)
    pi = [np.round(1 / k , decimals=8)] * k
    s = np.repeat([np.identity(2)], repeats=k, axis=0)

    return pi, m, s
