#!/usr/bin/env python3
"""
Convolution and pooling project
by Ced
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a convolution on grayscale images with custom padding:
    """
    
    (kh, kw) = kernel.shape
    # m is the number of images,  h is the height of  an image, w the width
    (m, h, w) = images.shape
    
    (ph, pw) = padding
    
    padded = np.pad(images, ((0, 0), (ph -1, ph - 1), (pw - 1, pw - 1)), mode='constant')
    print ("shape padded",ph, pw)


    conv_image = np.zeros((m, h + 2 * ph - 2, w + 2 * pw - 2))

    for i in range(h):
        for j in range(w):

            crop = padded[:, i: i + kh, j:j + kw]

            # beware to not sum the m number of images
            conv_image[:, i, j] = np.sum(crop[:] * kernel, axis=(1, 2))

    return conv_image