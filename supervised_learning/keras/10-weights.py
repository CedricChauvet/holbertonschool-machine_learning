#!/usr/bin/env python3
"""
this projet is about keras, learning with exercices, try a file
with only two methods
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    save the weights, using keras format
    """
    network.save_weights(filename)


def load_weights(network, filename):
    """
    load the weight
    """
    network.load_weights(filename)
