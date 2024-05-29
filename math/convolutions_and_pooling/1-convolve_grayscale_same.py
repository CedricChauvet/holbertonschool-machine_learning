#!/usr/bin/env python3
"""
Convolution and pooling project
by Ced
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a same convolution on grayscale images with padding
    """

    kh  = kernel.shape[0]
    kw = kernel.shape[1]
    

    # m is the number of images,  h is the height of  an image, w the width
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    # shape of the output
    conv_h = h - (kh - 1)
    conv_w = w - (kw - 1)

    padded = np.zeros((m, h + kw, w + kw))
    paddingw = (kw - 1)  // 2
    paddingh = (kh - 1)  // 2
    padded[:, paddingh:paddingh + h, paddingw:paddingw + w] = images
    # print("padded", images[0])
    # print("padding", padding)
    # creating array for the output
    conv_image = np.zeros((m, h, w))

    for i in range(conv_h):
        for j in range(conv_w):
            crop = padded[:, i: i + kh, j:j + kw]
            # beware to not sum the m number of images
            conv_image[:, i, j] = np.sum(crop[:] * kernel, axis=(1, 2))

    return conv_image

