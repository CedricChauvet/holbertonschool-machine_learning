#!/usr/bin/env python3
""" first program of Keras, to be continued"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ that builds a neural network with the Keras library
    be careful about  lmbtha and keep prob, with input style
    """

# inpuot of the model
    inputs = K.Input(shape=(nx,))
    i = 0
# Créating layerq
    x = inputs
    for lay, act in zip(layers, activations):
        i += 1
        x = K.layers.Dense(lay, activation=act,
                           kernel_regularizer=K.regularizers.l2(lambtha))(x)

        if i < len(layers):
            x = K.layers.Dropout(1 - keep_prob)(x)
    # crerating model, x is the last layers (considered ouput)
    model = K.Model(inputs=inputs, outputs=x)

    return model
