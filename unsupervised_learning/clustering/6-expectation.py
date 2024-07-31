#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    - X is a numpy.ndarray of shape (n, d) containing the data set
    - pi is a numpy.ndarray of shape (k,) containing the priors for each
    cluster
    - m is a numpy.ndarray of shape (k, d) containing
    the centroid means for each cluster
    - S is a numpy.ndarray of shape (k, d, d) containing
    the covariance matrices for each cluster
    return posterior an likelihood
    """
    n, d = X.shape
    k = pi.shape[0]

    g = np.zeros((k, n))
    sigma_g = np.zeros(n)

    for i in range(k):
        pdf_0 = pdf(X, m[i], S[i])
        g[i] = pi[i] * pdf_0
        sigma_g += g[i]

    # Calculer la vraisemblance totale (produit des probabilités)
    likelihood = np.sum(np.log(sigma_g))

    return g, likelihood
