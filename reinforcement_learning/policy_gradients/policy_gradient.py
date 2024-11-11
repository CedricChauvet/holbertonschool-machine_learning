#!/usr/bin/env python3
"""
This project is about policy gradient
By Ced
"""
import numpy as np


def policy(matrix, weight):
    """
    Function that computes to policy with a weight of a matrix
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z)
    return exp / np.sum(exp)