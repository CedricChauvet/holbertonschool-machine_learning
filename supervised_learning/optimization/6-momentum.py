#!/usr/bin/env python3
"""
Optimization project
by Ced
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    hat sets up the gradient descent with momentum optimization
    algorithm in TensorFlow
    return the optimizer
    """
    sgd = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    return sgd
