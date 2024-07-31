#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import numpy as np
from scipy.stats import multivariate_normal

def pdf(X, m, S):
    """
    Probability density function
    """
    #print("X shape", X.shape)
    pi = np.pi
    n, d = X.shape
    pdf = np.zeros(n)
    component_pdf = multivariate_normal(mean=m, cov=S).pdf(X)
    return component_pdf