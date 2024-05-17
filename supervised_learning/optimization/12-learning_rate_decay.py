#!/usr/bin/env python3
"""
Optimization project
Task 12 Learning Rate Decay Upgraded
by Ced
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    optim = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        name='ExponentialDecay'
)
    return optim