#!/usr/bin/env python3
"""
Project multivariate probability
By Ced+
"""
import numpy as np


class MultiNormal():
    """
    Multivariate Normal distribution
    first, get mean en covariance
    """

    def __init__(self, data):
        self.data = data
        self.n = data.shape[1]
        self.d = data.shape[0]

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        if self.n < 2:
            raise ValueError(" data must contain multiple data points")

        self.mean = np.reshape(np.mean(data.T, axis=0), ((self.d, 1)))
        self.cov = np.dot((data - self.mean),
                          (data - self.mean).T) / (self.n - 1)
