#!/usr/bin/env python3
"""
Bayesian Probability
"""
import numpy as np


def likelihood(x, n, P):
    """
    What is the likelihood ???
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that \
    is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if any(np.where(P > 1, True, False)) or any(np.where(P < 0, True, False)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # it s that!!
    vraisemblance = coef_binom(n, x) * np.power(P, x) * np.power(1 - P, n - x)
    return vraisemblance


def coef_binom(n, x):
    """
    computes the binomial coefficient
    """

    return np.math.factorial(n) /\
        (np.math.factorial(x) * np.math.factorial(n - x))
