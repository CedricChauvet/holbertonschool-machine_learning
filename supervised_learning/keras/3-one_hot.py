#!/usr/bin/env python3
"""Write a function def one_hot_encode(Y, classes): that converts
 a numeric label vector into a one-hot matrix:
"""
import tensorflow.keras as K


def one_hot(Y, classes=None):
    """transfom a numeric label vector into a one-hot matrix Ho is
     an np array an array of shape (Y, classes) with keras.utils.to_categorical
    """

    one_hot_matrix = K.utils.to_categorical(Y, num_classes=classes)
    return one_hot_matrix
