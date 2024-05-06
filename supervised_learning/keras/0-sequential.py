#!/usr/bin/env python3
import tensorflow.keras as K
""" first program of Keras, to be continued"""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ that builds a neural network with the Keras library"""

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
