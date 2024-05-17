#!/usr/bin/env python3
"""
Optimization project
Task 12 Learning Rate Decay Upgraded
by Ced
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate1, decay_step1):
    """
    creates a learning rate decay operation in tensorflow
    using inverse time decay:

    alpha is the original learning rate
    decay_rate is the weight used to determine the
    rate at which alpha will decay
    decay_step is the number of passes of gradient
    descent that should occur before alpha is decayed further
    the learning rate decay should occur in a stepwise fashion

    Returns: the learning rate decay operation
    """

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step1,
        decay_rate=decay_rate1,
        staircase=True,
        name='InverseTimeDecay'
    )
    return lr_schedule
