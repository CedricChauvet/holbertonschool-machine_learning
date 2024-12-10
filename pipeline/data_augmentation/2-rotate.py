#!/usr/bin/env python3
"""
pipeline data augmentation
Task 2 Rotate with tf.image
"""
import tensorflow as tf

def rotate_image(image):
    """
    * image is a 3D tf.Tensor containing the image to flip
    * Returns the flipped image
    """
    return tf.image.rot90(image, k=1)  # rotate image 90 degrees counter-clockwise