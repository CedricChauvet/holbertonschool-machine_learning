#!/usr/bin/env python3
""" first program of Keras, to be continued"""
import tensorflow.keras as K
from keras.optimizers import Adam


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


def optimize_model(network, alpha, beta1, beta2):
    """dkw"""
    optim = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=optim,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None
