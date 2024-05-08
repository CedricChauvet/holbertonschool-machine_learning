#!/usr/bin/env python3
"""
this projet is about keras, learning with exercices, try a file
with only two methods
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    network.save_weights(filename)

def load_weights(network, filename):
    network.load_weights(filename)


# layer_2.set_weights(layer_1.get_weights())