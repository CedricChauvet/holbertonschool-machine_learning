#!/usr/bin/env python3
"""
pipeline data augmentation
Task 5 change hue with tf.image
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    see this site for more info
    https://medium.com/@speaktoharisudhan/
    """
    return tf.image.adjust_hue(image, delta)
