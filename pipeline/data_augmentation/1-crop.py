#!/usr/bin/env python3
"""
pipeline data augmentation
Task 1 Crop with tf.image
"""
import tensorflow as tf


def crop_image(image, size):
    """
    still using tf.image, this time to crop the image    
    """
    height, width, channel = size

    cropped = tf.image.random_crop(image, size=[height, width, channel])
    return cropped
