#!/usr/bin/env python3
"""
pipeline data augmentation
Task 4 cbrightness with tf.image
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    change the brightness of an image with max_delta
    max_delta is the maximum amount the image should be brightened
    """
    return tf.image.adjust_brightness(image, delta=max_delta)
