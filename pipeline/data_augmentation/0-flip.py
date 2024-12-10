#!/usr/bin/env python3
"""
pipeline data augmentation
Task 0 Flip
"""
import tensorflow as tf


def flip_image(image):
    """
    * image is a 3D tf.Tensor containing the image to flip
    * Returns the flipped image
    """
    return tf.image.flip_left_right(image)  # flip image horizontally
