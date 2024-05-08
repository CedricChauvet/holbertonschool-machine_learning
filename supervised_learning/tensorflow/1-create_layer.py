#!/usr/bin/env python3
"""
first task on tensoflow project
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Task 1: Layers, using dense layers
    """
    l = tf.layers.dense(inputs=prev, units=n, activation=activation)
    return l
