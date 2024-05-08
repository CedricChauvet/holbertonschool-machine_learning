#!/usr/bin/env python3
"""
first task on tensoflow project
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    x = tf.placeholder(tf.float32, shape=(nx), name="x")
    y = tf.placeholder(tf.float32, shape=(classes), name="y")

    return x, y
