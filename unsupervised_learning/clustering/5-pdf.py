#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np


def pdf(X, m, S):
    """
    Probability density function
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or\
            len(S.shape) != 2 or S.shape[0] != S.shape[1]:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    n, d = X.shape
    pi = np.pi

    # Calcul de la constante de normalisation
    
    #determinant of covariance matrix
    det = np.linalg.det(S)
    norm_const = 1.0 / (np.power((2 * np.pi), d / 2) * np.sqrt(det))

    # Calcul de l'exponentielle
    
    # set the moyenne 
    x_mu = X - m
    # inverse of covariance matrix
    inv_cov = np.linalg.inv(S)

    result = np.exp(-0.5 * np.sum(np.dot(x_mu, inv_cov)
                                  * x_mu, axis=1))
    sol = norm_const * result
    # set minimum of sol values to 1e-300
    sol[sol < 1e-300] = 1e-300
    return sol
