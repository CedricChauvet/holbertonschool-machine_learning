#!/usr/bin/env python3
"""
This is the neural style Transfer
By Ced
"""
import tensorflow as tf
import numpy as np


class NST():
    """
    class modified during the advancement of the project:
    Neural Style Transfer
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        class constructor
        """

        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)  # the preprocessed style image
        self.content_image = self.scale_image(content_image)  # the preprocessed content image
        self.alpha = alpha  # the weight for content cost
        self.beta = beta  # the weight for style cost

    @staticmethod
    def scale_image(image):
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a\
                            numpy.ndarray with shape (h, w, 3)")

        h = image.shape[0]
        w = image.shape[1]

        if h >= w:
            w = int(w * 512 / h)
            h = 512
        else:
            h = int(h * 512 / w)
            w = 512

        # tensor = tf.zeros([1, h, w, 3])
        image = np.expand_dims(image, axis=0)
        image = tf.image.resize(image, [h, w], method='bicubic')
        image = image / 255.0
        image = tf.clip_by_value( image, 0, 1)


        return image
