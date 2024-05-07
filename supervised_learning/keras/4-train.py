#!/usr/bin/env python3
""" this projet is about keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose= False, shuffle=False):
    """
    this is the task 4, train a model
    """

    network.fit(data, labels, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=shuffle)

    return K.callbacks.History()