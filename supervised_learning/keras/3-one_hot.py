#!/usr/bin/env python3
"""Write a function def one_hot_encode(Y, classes): that converts
 a numeric label vector into a one-hot matrix:
"""
import tensorflow.keras as K


def one_hot(Y, classes=None):
    """transfom a numeric label vector into a one-hot matrix Ho is
     an np array of (cl,m), cl is the maximum number of classes found in Y,
    m the number of inputs"""
    
    classes = len(Y)
    Ho = [[0] * len(Y) for _ in range(classes)]
    for j, i in enumerate(Y):
        Ho[j][i] = 1
    return Ho