#!/usr/bin/env python3
"""
Optimization project
Task 12 Learning Rate Decay Upgraded
by Ced
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activ):
    """creates a batch normalization layer for a neural network in tensorflow:

    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used on the output of the layer
    you should use the tf.keras.layers.Dense layer as the base layer with kernal initializer tf.keras.initializers.VarianceScaling(mode='fan_avg')
    your layer should incorporate two trainable parameters, gamma and beta, initialized as vectors of 1 and 0 respectively
    you should use an epsilon of 1e-7
    
    Returns: a tensor of the activated output for the layer
    """




    # Calque d'entrée
    init = tf.keras.layers.Dense(n, 
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'))(prev)
    batchN = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones', )(init)
    activ = tf.keras.layers.Activation(activ)(batchN)
    return activ