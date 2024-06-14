#!/usr/bin/env python3
"""
Deep Convolutional Architectures project
by Ced
"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds a transition layer as described in DenseNet
    - X is the output from the previous layer
    - nb_filters is an integer representing the number of filters in X
    - compression is the compression factor for the transition layer
    Your code should implement compression as used in DenseNet-C
    """

    input0 = X

    BN0 = K.layers.BatchNormalization()(input0)
    ReLU0 = K.layers.Activation(activation='relu')(BN0)
    conv_0 = K.layers.Conv2D(
        filters=int(nb_filters * compression),
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=K.initializers.he_normal(seed=0),
        padding="same")(ReLU0)
    pool_0 = K.layers.AveragePooling2D(
        pool_size=(2, 2), strides=2, padding="same"
        )(conv_0)
    
    return pool_0, int(nb_filters * compression)
