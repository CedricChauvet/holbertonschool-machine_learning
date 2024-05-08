#!/usr/bin/env python3
"""
this projet is about keras, learning with exercices,
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    # evaluate the keras model

    loss, accuracy = network.evaluate(data, labels)
    return loss, accuracy
