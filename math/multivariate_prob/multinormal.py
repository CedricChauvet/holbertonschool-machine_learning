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

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        self.n = data.shape[1]
        self.d = data.shape[0]

        if self.n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.reshape(np.mean(data.T, axis=0), ((self.d, 1)))
        self.cov = np.dot((data - self.mean),
                          (data - self.mean).T) / (self.n - 1)

    def pdf(self, x):
        """
        Probability density function
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.d, 1):
            raise ValueError("x must have the shape ({self.d}, 1)")
        pi = np.pi
        pdf_first = 1 / (pow(2 * pi, self.d / 2) *
                         pow(np.linalg.det(self.cov), 1 / 2)) *\
            np.exp(-1 / 2 * (x - self.mean).T @
                   np.linalg.inv(self.cov) @ (x - self.mean))

        return float(pdf_first)
