#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Calculate the centroid by K mean algorithm
    return  the K centroids and the clss
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    centroid = np.random.uniform(low=np.min(X, axis=0),
                                 high=np.max(X, axis=0), size=(k, d))

    for i in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroid, axis=2)
        clss = np.argmin(distances, axis=1)

        new_centroid = np.copy(centroid)

        for j in range(k):
            if len(np.where(clss == j)[0]) == 0:
                # print("on a pas  trouvé", j) dans clss
                centroid[j] = np.random.uniform(np.min(X, axis=0),
                                                np.max(X, axis=0), d)
            else:
                # print("on a ", j) on a trouvé j dans clsss
                centroid[j] = np.mean(X[np.where(clss == j)], axis=0)

        if np.array_equal(new_centroid, centroid):
            break

    return centroid, clss
