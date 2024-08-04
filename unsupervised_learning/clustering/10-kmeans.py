#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import sklearn.cluster


def kmeans(X, k):
    """
    get kmean with sklearn
    return centroids and labels
    """

    kmean = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmean.cluster_centers_
    clss = kmean.labels_

    return C, clss
