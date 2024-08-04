#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):

    n, d = X.shape
    l_arr = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)
    for k in range(kmin, kmax + 1):
        print("K", k)
        pi, m, S, g, l=  expectation_maximization(X, k, iterations, tol, verbose=False)
       
        b[k - kmin] = k * np.log(n) - (2 * l)
        l_arr[k- kmin] =l
    best_k = np.argmax(b) + kmin
    best_l = np.argmax(l_arr)
    print("best k", best_k)

    #expectation_maximization(X, 1, iterations=50, verbose=True) 
    best_result = pi, m, S
    print("best result", best_result)
    return best_k, best_result, l, b