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

    # padding is exactly pw and pw in each sides
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # see the vidéo of deeplearningIA for explication
    # of the size of conv_images
    conv_image = np.zeros((m, h + 2 * ph - kh + 1, w + 2 * pw - kw + 1))

    for i in range(h + 2 * ph - kh + 1):
        for j in range(w + 2 * pw - kw + 1):
            crop = padded[:, i: i + kh, j:j + kw]

            # beware to not sum the m number of images
            conv_image[:, i, j] = np.sum(crop[:] * kernel, axis=(1, 2))

    return conv_image
