#!/usr/bin/env python3
"""
first task on tensoflow project
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Task 1: Layers, using dense layers
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    l = tf.layers.dense(inputs=prev, units=n, activation=activation,kernel_initializer=initializer)
    return l
