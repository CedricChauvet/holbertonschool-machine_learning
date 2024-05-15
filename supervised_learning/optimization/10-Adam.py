#!/usr/bin/env python3
"""
Optimization project
Task 10 Adam upgraded with tensorflow
by Ced
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Write the function def create_Adam_op(alpha, beta1, beta2, epsilon):
    that sets up the Adam optimization algorithm in TensorFlow:

    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero

    Returns: optimizer
    """

    optim = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon,
        name='adam',
    )

    return optim
