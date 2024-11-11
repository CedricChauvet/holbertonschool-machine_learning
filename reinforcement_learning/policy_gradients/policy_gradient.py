#!/usr/bin/env python3
"""
This project is about policy gradient
By Ced
"""
import numpy as np


def policy(matrix, weight):
    """
    Function that computes policy with a weight of a matrix
    weight: matrix of random weight, my policy
    matrix: state or observation of the environment
    returns softamx policy
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z)
    return exp / np.sum(exp)
