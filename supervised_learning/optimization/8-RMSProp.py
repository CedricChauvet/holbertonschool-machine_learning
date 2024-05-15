#!/usr/bin/env python3
"""
Optimization project
Task 8 RMSProp upgraded with tensorflow
by Ced
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    sets up the RMSProp optimization algorithm in TensorFlow

    alpha is the learning rate
    beta2 is the RMSProp weight (Discounting factor)
    epsilon is a small number to avoid division by zero
    Returns: optimizer
    """

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon,
        name='rmsprop',
    )

    return optimizer
