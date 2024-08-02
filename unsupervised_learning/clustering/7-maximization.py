#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np


def maximization(X, g):
    """
    but what does maximization do?
    je crois qu'elle ecarte les clusters
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    pi = np.zeros(k)
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        pi[i] = 1 / n * np.sum(g[i], axis=0)
        m = np.sum(g[:, :, np.newaxis] * X, axis=1) /\
            np.sum(g, axis=1)[:, np.newaxis]
        S[i] = (g[i] * (X - m[i]).T @ (X - m[i])) / np.sum(g[i], axis=0)

    return pi, m, S
