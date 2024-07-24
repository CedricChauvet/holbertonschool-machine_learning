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

    d = C.shape[0]

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[1] != d or len(C.shape) != 2:
        raise ValueError(" C must be a 2D square matrix")
    
    corr_mat = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            corr_mat[i,j] =  C[i][j] / np.sqrt(C[i][i] * C[j][j])
    
    return corr_mat
