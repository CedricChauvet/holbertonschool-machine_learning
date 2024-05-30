#!/usr/bin/env python3
"""
Convolution and pooling project
by Ced
"""
import numpy as np

def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a same convolution on grayscale images with padding
    """

    (kh, kw) = kernel.shape
    # m is the number of images,  h is the height of  an image, w the width
    (m, h, w) = images.shape
    (sh, sw) = stride
    
    
    
    if padding == 'valid':    
        # shape of the output
       ph = 0
       pw = 0
       conv_h = ((h + 2 * ph - kh) / sh + 1) // 1
       conv_w = ((w + 2 * pw - kw) / sw + 1) // 1

    elif padding == 'same':
        
        conv_h = kh // 2 //stride[0]
        conv_w = kw // 2 //stride[1]
    

    elif  type(padding) is tuple:
 
        conv_h = ((h + 2 * ph - kh) / sh + 1) // 1
        conv_w = ((w + 2 * pw - kw) / sw + 1) // 1





    # creating array for the output
    conv_image = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            crop = images[:, sh * i: sh * i + kh, sw * j: sw * j + kw]
            # beware to not sum the m number of images
            conv_image[:, i, j] = np.sum(crop[:] * kernel, axis=(1, 2))

    return conv_image




