#!/usr/bin/env python3
"""
Project multivariate probability
By Ced+
"""
import numpy as np


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
    for i in range(d):
        mean[0, i] = np.sum(X[:, i]) / n

    cov = np.dot(X.T, X) / n - np.dot(mean.T, mean)

    return mean, cov
