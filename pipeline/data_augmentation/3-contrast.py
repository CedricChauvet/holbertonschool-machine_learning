#!/usr/bin/env python3
"""
pipeline data augmentation
Task 3 contrast with tf.image
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    contrast image using tf.image
    """
    contrast = tf.random.uniform((), lower, upper)
    return tf.image.adjust_contrast(image, contrast_factor=contrast)
