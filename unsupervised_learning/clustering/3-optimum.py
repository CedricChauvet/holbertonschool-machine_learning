#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    analyze variance wrt numbers of clusters
    """


    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    n, d1 = X.shape

    if not isinstance(kmin, int) and not isinstance(kmax, int) and kmax < kmin:
        return None
    
    if not isinstance(kmax, int) or kmax < 0:
        return None

    
    if not isinstance(kmin, int) or kmax < 0:
        return None
    
    n, d = X.shape
    results = []
    d_var = []

    if kmax is None or kmax >= n:
        kmax = n


    for k in range(kmin, kmax+ 1 ):
        # initialize random k-centroids
        centroid = np.random.uniform(low=np.min(X, axis=0),
                                        high=np.max(X, axis=0), size=(k, d))

        for i in range(iterations):

            distances = np.linalg.norm(X[:, np.newaxis] - centroid, axis=2)
            clss = np.argmin(distances, axis=1)

            new_centroid = np.copy(centroid)

            for j in range(k):
                # new centroid
                if len(np.where(clss == j)[0]) == 0:
                    centroid[j] = np.random.uniform(np.min(X, axis=0),
                                                    np.max(X, axis=0), d)
                
                else:
                    centroid[j] = np.mean(X[np.where(clss == j)], axis=0)
            # if centroid don't change, break
            if np.array_equal(new_centroid, centroid):
                break

        if k == kmin:
            var_min = var
        var = np.sum((X[:] - centroid[clss]) ** 2)
        results.append((centroid, clss))
        
        d_var.append(var_min - var)

    return results, d_var
