#!/usr/bin/env python3
"""
Project Hiden Markov Model
By Ced+
"""
import numpy as np


def regular(P):
    """
    get transition matrice times a lot
    """

    n, n = P.shape
    t = 300
    s = np.zeros((1, n))
    s[0, 0] = 1
    limit = np.linalg.matrix_power(P, t)

    if np.any(limit == 0):
        return None

    else:
        return s @ limit
