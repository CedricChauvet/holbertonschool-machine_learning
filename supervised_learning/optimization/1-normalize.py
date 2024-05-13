#!/usr/bin/env python3
"""
Optimization project
by Ced
"""
import numpy as np


def normalize(X, m, s):
    """
    task 1: Normalize
    this function normalizes (standardizes) a matrix
    """
    # we center the data on the origin
    mean = X - m
    # stretching!
    stand = mean / s

    return stand
