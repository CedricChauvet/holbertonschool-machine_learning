#!/usr/bin/env python3
import numpy as np
""""
Task 13, linear algebre project: . Cat's Got Your Tongue
"""


def np_cat(mat1, mat2, axis=0):
    """ assume concatenation on 2 matrices along a specific axe """
    if axis == 0:
        """ sum de 0 dim of the matrix,t the concatenate on this axe"""
        mat_cat = np.concatenate((mat1, mat2), axis=axis)
        return mat_cat
