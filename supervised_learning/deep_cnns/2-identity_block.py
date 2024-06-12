#!/usr/bin/env python3
"""
Deep Convolutional Architectures project
by Ced
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    to be defined
    """

    (F11, F3, F12) = filters

    conv2D = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same")(A_prev)
    BN1 = K.layers.BatchNormalization()(conv2D)
    ReLU1 = K.layers.Activation(activation='relu')(BN1)

    conv2D_1 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same")(ReLU1)
    BN2 = K.layers.BatchNormalization()(conv2D_1)
    ReLU2 = K.layers.Activation(activation='relu')(BN2)

    conv2D_2 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same")(ReLU2)
    BN2 = K.layers.BatchNormalization()(conv2D_2)

    add = K.layers.Add()([BN2, A_prev])
    ReLU3 = K.layers.Activation(activation='relu')(add)

    return ReLU3
