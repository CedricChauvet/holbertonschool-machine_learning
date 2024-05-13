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
    m = X.shape[0]
    nx = X.shape[1]
    ny = Y.shape[1]
    matX_Y = np.concatenate((X, Y), axis=1)

    mat_sh = np.random.permutation(matX_Y)
    X_sh = mat_sh[:, 0:nx]
    Y_sh = mat_sh[:, nx:]

    return X_sh, Y_sh
