#!/usr/bin/env python3
"""
Deep Convolutional Architectures project
by Ced
"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    input0 = X

        
    for i in range(layers):
        BN0 = K.layers.BatchNormalization()(input0)
        ReLU0 = K.layers.Activation(activation='relu')(BN0)
        conv_0 = K.layers.Conv2D(
            filters = 4 * growth_rate,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=K.initializers.he_normal(seed=0),
            padding="same")(ReLU0) 
        
        BN1 = K.layers.BatchNormalization()(conv_0)
        ReLU1 = K.layers.Activation(activation='relu')(BN1)
        conv_1 = K.layers.Conv2D(
            filters= growth_rate,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=K.initializers.he_normal(seed=0),
            padding="same")(ReLU1)
        input0 = K.layers.Concatenate()([input0, conv_1,])

    return input0, nb_filters + layers * growth_rate
