#!/usr/bin/env python3
"""
3rd task on tensoflow project
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    """
    Write the function def calculate_loss(y, y_pred):
    that calculates the softmax cross-entropy loss of a prediction:

    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the loss of the prediction
    """
    loss = tf.compat.v1.losses.softmax_cross_entropy(
           onehot_labels=y, logits=y_pred)

    return loss
