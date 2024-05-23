#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import tensorflow as tf


class CustomDropoutLayer(tf.keras.layers.Layer):
    """
    creating a class in order to call training
    """
    def __init__(self, rate, train):
        super(CustomDropoutLayer, self).__init__()
        self.rate = rate
        self.dropout = tf.keras.layers.Dropout(rate)
        self.train = train

    def call(self, inputs, training):
        return self.dropout(inputs, training=self.train)

def dropout_create_layer(prev, n, activ, keep_prob, training=True):
    """
    create a layer with dropout
    """
    x = tf.keras.layers.Dense(n, activation=activ)(prev)

    # Custom Dropout layer with explicit training argument
    x = CustomDropoutLayer(1 - keep_prob, training)(x)
    return x
