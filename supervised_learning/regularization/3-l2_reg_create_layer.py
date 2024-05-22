#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer with L2 regularization.

    Args:
        prev: Tensor containing the output of the previous layer.
        n: Number of nodes for the new layer.
        activation: Activation function to be used on the layer.
        lambtha: L2 regularization parameter.

    Returns:
        Output tensor of the new layer.
    """
    # Create a dense layer with L2 regularization
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )(prev)

    return layer








"""
def l2_reg_create_layer(prev, n, activ, lambtha):
    


    L2_layer = tf.keras.layers.Dense(units=n,activation=activ,
           kernel_regularizer=tf.keras.regularizers.L2(
    l2=lambtha))(prev)
    
    return L2_layer    


#kernel_regularizer='l2', activity_regularizer='l2'
"""