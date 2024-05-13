#!/usr/bin/env python3
"""
Optimization project

"""
import numpy as np


def normalization_constants(X):
    """
    Do the mean and standard deviation of each feature (here a, b, c)
    the normalization (standardization) constants of a matrix
    """
    # n is the length of element to process
    n = X.shape[0]

    # using list to compute
    m = []
    s = []
    for elmt in X.T:  # beware of transpose

        # this is the mean
        m.append(np.sum(elmt)/n)
        # The standard deviation,
        # usually called "Population Standard Deviation"
        s.append(np.sqrt(np.sum(1 / n * np.sum((elmt - m[-1]) ** 2))))
    return m, s
