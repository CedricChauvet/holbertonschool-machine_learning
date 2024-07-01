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
        
        
        if not isinstance(style_image,np.ndarray) or style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image,np.ndarray) or content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
    
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        
        self.style_image = self.scale_image(style_image) #  the preprocessed style image
        self.content_image = self.scale_image(content_image) # the preprocessed content image
        self.alpha = alpha # the weight for content cost
        self.beta = beta #  the weight for style cost


    @staticmethod
    def scale_image(image):
        if not isinstance(image,np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        

        if image.shape[0] > 512 or image.shape[1] > 512:
            resized_image = tf.image.resize(image, [512, 512], preserve_aspect_ratio=True)

        rescaled_image =  resized_image / 255        
        reshaped_image = tf.expand_dims(rescaled_image,axis=0)
        return reshaped_image
        