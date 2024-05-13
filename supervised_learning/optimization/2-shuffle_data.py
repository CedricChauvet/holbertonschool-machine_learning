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
    # print("m , nx , ny", m ,nx ,ny)
    X_sh = np.random.permutation(X).reshape((m, nx))
    Y_sh = np.random.permutation(Y).reshape((m, ny))

    return X_sh, Y_sh
