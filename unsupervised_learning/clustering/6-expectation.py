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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    n, d = X.shape
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    k = pi.shape[0]

    if k <= 0 or not np.isclose(np.sum(pi), 1):
        return None, None

    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3 or\
            S.shape[1] != S.shape[2]:
        return None, None

    if m.shape[1] != d or S.shape[2] != d:
        return None, None

    if pi.shape[0] != k or S.shape[0] != k or m.shape[0] != k:
        return None, None

    g = np.zeros((k, n))
    sigma_g = np.zeros(n)

    for i in range(k):
        pdf_0 = pdf(X, m[i], S[i])
        g[i] = pi[i] * pdf_0
        sigma_g += g[i]

    g = g / sigma_g

    # Calculer la vraisemblance totale (produit des probabilités)
    likelihood = np.sum(np.log(sigma_g))

    return g, likelihood
