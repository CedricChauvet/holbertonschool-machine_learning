#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    combition of  task expectation and optimization the achive GMM
    """

    pi, m, S = initialize(X, k)
    new_L = 0
    for i in range(iterations):

        g, L = expectation(X, pi, m, S)


        if verbose and (i % 10 == 0):
            print(f"Log Likelihood after {i} iterations: {L:.5f}")

        if abs(L - new_L) < tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: {L:.5f}")
            break
        new_L=L    
        pi, m, S = maximization(X, g)

    # g, L = expectation(X, pi, m, S)

    return pi, m, S, g, L
