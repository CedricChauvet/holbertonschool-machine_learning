#!/usr/bin/env python3
"""
Optimization project
by Ced
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Write the functionthat shuffles the data points
    in two matrices the same way:
    """

    X_sh = np.random.permutation(X)
    Y_sh = np.random.permutation(Y)

    return X_sh, Y_sh
