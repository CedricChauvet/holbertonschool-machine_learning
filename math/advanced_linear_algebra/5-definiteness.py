#!/usr/bin/env python3
"""
Project linear algebra
By Ced
"""
import numpy as np


def definiteness(matrix):
    """
    produce the definiteness of matrice
    by calculating eigen values
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2:
        return None
    if (matrix.shape[0] != matrix.shape[1]):
        return None

    eigenvalues, _ = np.linalg.eig(matrix)
    if all(eigenvalues >= 0) and any(eigenvalues == 0):
        return "Positive semi-definite"
    if all(eigenvalues <= 0) and any(eigenvalues == 0):
        return "Negative semi-definite"
    if all(eigenvalues > 0):
        return "Positive definite"
    if all(eigenvalues < 0):
        return "Negative definite"
    if any(eigenvalues > 0) and any(eigenvalues < 0):
        return "Indefinite"
