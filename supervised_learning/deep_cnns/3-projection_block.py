#!/usr/bin/env python3
"""
Deep Convolutional Architectures project
by Ced
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    computes a projection block,
    - A_prev is the input layer
    - filters are the size of cnn
    - s is the stride of the first convolution
    in both the main path and the shortcut connection
    returns: the activated output of the projection block
    """

    (F11, F3, F12) = filters
    # module with conv 2D layer followed by batch normalization
    # and activation
    conv2D = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        kernel_initializer=K.initializers.he_normal(seed=0),
        padding="same")(A_prev)
    BN1 = K.layers.BatchNormalization()(conv2D)
    ReLU1 = K.layers.Activation(activation='relu')(BN1)

    conv2D_1 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer=K.initializers.he_normal(seed=0),
        padding="same")(ReLU1)
    BN2 = K.layers.BatchNormalization()(conv2D_1)
    ReLU2 = K.layers.Activation(activation='relu')(BN2)

    conv2D_2 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=K.initializers.he_normal(seed=0),
        padding="same")(ReLU2)
    BN3 = K.layers.BatchNormalization()(conv2D_2)

    shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        kernel_initializer=K.initializers.he_normal(seed=0),
        padding="same")(A_prev)
    BN4 = K.layers.BatchNormalization()(shortcut)

    add = K.layers.Add()([BN3, BN4])

    ReLU3 = K.layers.Activation(activation='relu')(add)

    return ReLU3
