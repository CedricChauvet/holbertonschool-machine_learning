#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np
initialize = __import__('0-initialize').initialize



def kmeans(X, k, iterations=1000):
    
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    centroid = initialize(X, k)

    n, d = X.shape
    C = np.zeros((k, d))
    C_sum = np.zeros((n, k,d))
    clss = np.zeros(n)
    point = np.ones(n)
    for i in range(n):
        dist = []
        for j in range(k):
            print("X", X[i])
            print("centroid", centroid[j])
            dis = distance(X[i],centroid[j])
            dist.append(dis)

        clss[i] = np.argmin(dist)
        C_sum[i, int(clss[i])] = X[i]

    C = C_sum.mean(axis =0)
    print("cent", C.shape) 
    return C, clss


def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))
