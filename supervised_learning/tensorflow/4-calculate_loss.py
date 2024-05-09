#!/usr/bin/env python3
"""
3rd task on tensoflow project
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    loss = tf.compat.v1.losses.softmax_cross_entropy(
           onehot_labels=y, logits=y_pred)
    return loss
