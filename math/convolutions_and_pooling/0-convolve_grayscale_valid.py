#!/usr/bin/env python3
"""
Convolution and pooling project
by Ced
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images:

    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images

    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images

    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    """

    # print("kern shape",kernel.shape[0])
    kw = kernel.shape[0]
    # m is the number of images,  h is the height of  an image, w the width
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    # shape of the output
    conv_h = h - (kw - 1)
    conv_w = w - (kw - 1)

    # creating array for the output
    conv_image = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            crop = images[:, i: i + kw, j:j + kw]
            # beware to not sum the m number of images
            conv_image[:, i, j] = np.sum(crop[:] * kernel, axis=(1, 2))

    return conv_image
