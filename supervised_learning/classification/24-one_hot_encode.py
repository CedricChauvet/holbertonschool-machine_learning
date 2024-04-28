#!/usr/bin/env python3

import numpy as np

"""Write a function def one_hot_encode(Y, classes): that converts
 a numeric label vector into a one-hot matrix:
"""


def one_hot_encode(Y, classes):
    """transfom a numeric label vector into a one-hot matrix Ho is
     an np array of (cl,m), cl is the maximum number of classes found in Y,
    m the number of inputs"""
   
    if type(Y) is not np.ndarray or type(classes) is not int or classes < 2 or classes < np.max(Y):
        return None

    Ho = np.zeros((classes, Y.shape[0]))
    for i, j in enumerate(Y):
        Ho[j, i] = 1
    return Ho
