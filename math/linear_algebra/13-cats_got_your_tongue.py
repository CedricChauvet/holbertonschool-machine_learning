#!/usr/bin/env python3
""""
Task 13, linear algebre project: . Cat's Got Your Tongue
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ assume concatenation on 2 matrices along a specific axe """
    mat_cat = np.concatenate((mat1, mat2), axis=axis)
    return mat_cat
