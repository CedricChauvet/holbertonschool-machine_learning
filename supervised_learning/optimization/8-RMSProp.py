#!/usr/bin/env python3
"""
Optimization project
Task 8 RMSProp upgrader with tensorflow
by Ced
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=alpha,
    rho=beta2,
    epsilon=epsilon,
    name='rmsprop',
    )
    return optimizer