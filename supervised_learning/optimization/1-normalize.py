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

    mean = X - m
    stand = mean / s
    return stand