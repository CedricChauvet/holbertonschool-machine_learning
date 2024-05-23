#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activ, keep_prob, training=True):
    """
    Create a layer of a neural network using dropout.

    Args:
        prev: A tensor containing the output of the previous layer.
        n: The number of nodes the new layer should contain.
        activation: The activation function for the new layer.
        keep_prob: The probability that a node will be kept.
        training: A boolean indicating whether the model is in training mode.

    Returns:
        The output of the new layer.
    """
    # Create a dense layer

    # got initialyzer on task1 tensorflow projet
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(units=n, activation=activ, kernel_initializer=initializer)(prev)
    
    # Apply dropout
    dropout_layer = tf.keras.layers.Dropout(1 - keep_prob, seed=4)(dense_layer, training=training)
    
    return dropout_layer