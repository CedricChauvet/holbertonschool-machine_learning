#!/usr/bin/env python3
"""
Convolution and pooling project
by Ced
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Computes a pooling layer,

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the
    kernel for the pooling
    stride is a tuple of (sh, sw) containing the strides for the pooling
    mode is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively

    Returns: the output of the pooling layer
    """
    # A_prev is the input of the convolution layer
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    # print("Al-1", m,h_prev,w_prev,c_prev)

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    # define stride
    sh = stride[0]
    sw = stride[1]

    # define the shape of output, must be a numpy
    out_h = (h_prev - kh) // sh + 1
    out_w = (w_prev - kw) // sw + 1
    out_c = A_prev.shape[3]

    A_pool = np.zeros((m, out_h, out_w, out_c))

    for i in range(out_h):
        for j in range(out_w):
            for c in range(out_c):
                x = A_prev[:, sh * i: sh * i + kh, sw * j: sw * j + kw, c]
                if mode == "max":
                    x = np.max(x, axis=(1, 2))
                if mode == "avg":
                    x = np.mean(x, axis=(1, 2))
                A_pool[:, i, j, c] = x

    return A_pool
