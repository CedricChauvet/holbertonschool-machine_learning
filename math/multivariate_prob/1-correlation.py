#!/usr/bin/env python3
"""
Project multivariate probability
By Ced+
"""
import numpy as np


def correlation(C):
    """
    calcul the correlation matrice give the convolution
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[1] != C.shape[0]  :
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]

    corr_mat = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            corr_mat[i, j] = C[i][j] / np.sqrt(C[i][i] * C[j][j])

    return corr_mat
