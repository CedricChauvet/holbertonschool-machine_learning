#!/usr/bin/env python3
"""
Optimization project
Task 12 Learning Rate Decay Upgraded
by Ced
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate1, decay_step1):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step1,
        decay_rate=decay_rate1)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    return optimizer