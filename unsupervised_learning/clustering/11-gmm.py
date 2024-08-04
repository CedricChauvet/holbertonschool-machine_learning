#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import sklearn.mixture


def gmm(X, k):
    """
    give Gaussian mixture using scikit
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    m = gmm.means_
    pi = gmm.weights_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)

    return pi, m, S, clss, bic
