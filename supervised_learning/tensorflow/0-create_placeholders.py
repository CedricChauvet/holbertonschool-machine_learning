#!/usr/bin/env python3
"""
first task on tensoflow project
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Zero task: 0. Placeholders
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
