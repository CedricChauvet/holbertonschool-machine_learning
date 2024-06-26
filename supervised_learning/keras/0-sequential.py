#!/usr/bin/env python3
""" first program of Keras, to be continued"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ that builds a neural network with the Keras library
    be careful about  lmbtha and keep prob, that ensur the dropout
    """

    # define the keras model
    model = K.Sequential()
    input_shape = (nx,)
    i = 0

    for lay, act in zip(layers, activations):
        i += 1
        model.add(K.layers.Dense(lay, activation=act, input_shape=(nx,),
                  kernel_regularizer=K.regularizers.l2(lambtha)))
        if i < len(layers):
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
