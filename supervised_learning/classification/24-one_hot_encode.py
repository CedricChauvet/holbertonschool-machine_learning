#!/usr/bin/env python3

import numpy as np

"""Write a function def one_hot_encode(Y, classes): that converts
 a numeric label vector into a one-hot matrix:
"""


def one_hot_encode(Y, classes):
    """transfom a numeric label vector into a one-hot matrix Ho is
     an np array of (cl,m), cl is the maximum number of classes found in Y,
    m the number of inputs"""
    Ho = np.zeros((10, Y.shape[0]))


    for i, j in enumerate(Y):
        Ho[j, i] = 1
    return Ho
