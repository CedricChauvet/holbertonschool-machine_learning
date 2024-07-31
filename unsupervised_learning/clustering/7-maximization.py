#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np


def maximization(X, g):
    """
    but what does maximization do?
    """
    n, d = X.shape
    k = g.shape[0]
    print("X = ", X.shape)
    print("g = ", g.shape)

    pi = np.zeros(k)
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        pi[i] = 1/ n * np.sum(g[i], axis=0)
        m = np.sum(g[:, :, np.newaxis] * X, axis=1) / np.sum(g, axis=1)[:, np.newaxis]
        #m[i,:] = np.sum(g[i,np.newaxis] * X,axis =0) / np.sum(g[i])
        #m[i][0] = np.sum(g[i] * X[:, 0]) / np.sum(g[i])
        #m[i][1] = np.sum(g[i] * X[:, 1]) / np.sum(g[i])
    # print("pi", pi)
    print("mu", m)
