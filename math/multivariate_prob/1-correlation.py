#!/usr/bin/env python3
"""
Project multivariate probability
By Ced+
"""
import numpy as np


def correlation(C):
    d = C.shape[0]

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[1] != d or len(C.shape) != 2:
        raise ValueError(" C must be a 2D square matrix")
    




def mean_cov(X):
    """
    gives mean and covariance in a matrix
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    n = X.shape[0]
    d = X.shape[1]
    mean = np.zeros((1, d))
    cov = np.zeros((d, d))

    mean[0] = np.mean(X, axis=0)

    cov = np.dot((X - mean).T, (X - mean)) / (n - 1)

    return mean, cov
