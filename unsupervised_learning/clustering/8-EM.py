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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    
    # Initialize the parameters
    pi, m, S = initialize(X, k)
    prev_L = None
    
    for i in range(iterations):
        # E-step: calculate the responsibilities and the log likelihood
        g, L = expectation(X, pi, m, S)

        # Verbose logging
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {L:.5f}")

        # Check for convergence
        if prev_L is not None and abs(L - prev_L) <= tol:
            if verbose:
                print(f"Log Likelihood converged after {i} iterations: {L:.5f}")
            break

        # Update the previous log likelihood
        prev_L = L
        
        # M-step: update the parameters
        pi, m, S = maximization(X, g)

    # Final verbose logging
    if verbose and (i + 1) % 10 != 0:
        print(f"Log Likelihood after {i + 1} iterations: {L:.5f}")

    return pi, m, S, g, L
