#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np
import sklearn.cluster


def kmeans(X, k):
    kmean = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmean.cluster_centers_
    clss = kmean.labels_

    return C, clss
