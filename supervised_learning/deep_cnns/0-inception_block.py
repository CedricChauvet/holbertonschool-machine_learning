#!/usr/bin/env python3
"""
Deep Convolutional Architectures project
by Ced
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    do a simple inception network
    """
    (F1, F3R, F3, F5R, F5, FPP) = filters

    conv2d = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation=K.layers.ReLU(),
        kernel_initializer=K.initializers.he_normal(seed=0),
        )(A_prev)

    conv2d_1 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation=K.layers.ReLU(),
        kernel_initializer=K.initializers.he_normal(seed=0),
        )(A_prev)

    conv2d_2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation=K.layers.ReLU(),
        kernel_initializer=K.initializers.he_normal(seed=0)
        )(conv2d_1)

    conv2d_3 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation=K.layers.ReLU(),
        kernel_initializer=K.initializers.he_normal(seed=0),
        )(A_prev)

    conv2d_4 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation=K.layers.ReLU(),
        kernel_initializer=K.initializers.he_normal(seed=0)
        )(conv2d_3)

    max_pooling2d = K.layers.MaxPooling2D(
        strides=1, padding="same"
        )(A_prev)

    conv2d_5 = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation=K.layers.ReLU(),
        kernel_initializer=K.initializers.he_normal(seed=0),
        )(max_pooling2d)

    OUT = K.layers.Concatenate()([conv2d, conv2d_2, conv2d_4, conv2d_5])

    return OUT
