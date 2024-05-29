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

    kh = kernel[:,0].shape[0]
    kw = kernel[0,:].shape[0]
    print("kernel", kh , kw)   

    # m is the number of images,  h is the height of  an image, w the width
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    paddingh = (kh - 1)  // 2
    paddingw = (kw - 1)  // 2
    

    padded = np.zeros((m, h +  2* paddingh , w + 2 * paddingw))
    padded[:, paddingh:paddingh + h, paddingw:paddingw + w] = images

    # print("padding", padding)
    # creating array for the output
    conv_image = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            crop = padded[:, i: i + kh, j:j + kw]
            # beware to not sum the m number of images
            conv_image[:, i, j] = np.sum(crop[:] * kernel, axis=(1, 2))
    #print("conv image", conv_image)
    return conv_image

