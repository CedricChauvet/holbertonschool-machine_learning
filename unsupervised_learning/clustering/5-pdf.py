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
    n, d = X.shape
    pi = np.pi

    # Calcul de la constante de normalisation
    det = np.linalg.det(S)
    norm_const = 1.0 / (np.power((2 * np.pi), d / 2) * np.sqrt(det))

    # Calcul de l'exponentielle
    x_mu = X - m
    inv_cov = np.linalg.inv(S)
    result = np.exp(-0.5 * np.sum(np.dot(x_mu, inv_cov) * x_mu, axis=1))

    return norm_const * result
