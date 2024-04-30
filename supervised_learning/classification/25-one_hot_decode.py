#!/usr/bin/env python3
"""Write a function def one_hot_encode(Y, classes): that converts
 a numeric label vector into a one-hot matrix and reverse
 Task 25: One-Hot Decode
"""
import numpy as np


def one_hot_encode(Y, classes):
    """transfom a numeric label vector into a one-hot matrix Ho is
     an np array of (cl,m), cl is the maximum number of classes found in Y,
    m the number of inputs"""

    if type(Y) is not np.ndarray or type(classes) is not int or classes < 2 \
            or classes < np.max(Y):
        return None

    Ho = np.zeros((classes, Y.shape[0]))
    for i, j in enumerate(Y):
        Ho[j, i] = 1
    return Ho


def one_hot_decode(one_hot):
    """converts a one-hot matrix into a vector of labels"""

    if one_hot is None:
        return None
    if type(one_hot) is not np.ndarray:
        return None

    try:
        m = one_hot.shape[1]
        classes = one_hot.shape[0]
        Ho_De = np.ones((1, m), dtype=int)
        for index, i in enumerate(one_hot.T):
            Ho_De[0, index] = (np.where(i == 1)[0][0])

        return Ho_De[0]

    except Exception as ex:
        return None
