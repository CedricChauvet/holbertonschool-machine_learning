#!/usr/bin/env python3
"""
pipeline data augmentation
Task 1 Crop with tf.image
"""
import tensorflow as tf


def crop_image(image, size):
    """
    
    """
    height, width, _ = size
    print("hg", height, width)
    cropped = tf.keras.layers.RandomCrop( 200, 200)(image)  # crop image
    
    print("cropped", cropped)