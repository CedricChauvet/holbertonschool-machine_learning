#!/usr/bin/env python3
"""
Convolution and pooling project
by Ced
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
        """
        -  dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) 
        containing the partial derivatives
        -  A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        -  W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
        containing the kernels for the convolution
        -  b is a numpy.ndarray of shape (1, 1, 1, c_new)
        containing the biases applied to the convolution
        -  padding is a string that is either same or valid
        -  stride is a tuple of (sh, sw) containing the strides
        
        Returns: the partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db), respectively
        """
        
        print("dZ", dZ.shape)
        print("A_prev", A_prev.shape)
        print("W", W.shape)
        print("b", b.shape)
        
        (m, h_new, w_new, c_new) = dZ.shape
        (m, h_prev, w_prev, c_prev) = A_prev.shape
        (kh, kw, c_prev, c_new) = W.shape
        (sh, sw) = stride
        
        if padding == 'valid':
            ph = 0
            pw = 0
        if padding == 'same':
            ph = int((h_prev * (sh - 1) + kh - sh) / 2)
            pw = int((w_prev * (sw - 1) + kw - sw) / 2)

        db = np.sum(dZ, axis=(0,1,2))
        print("db",db) # ok
        
        for f in range (c_new):
            for c in range(c_prev):
                for k in range(kh):
                    for l in range(kw):
                        # attention au signe * ci dessous
                        dW[f,c,k,l] = dZ[:, :, :, f] * A_prev[:, : + k - 1, : + l -1, c]
                        
                        
        
        
        print("dw", dw.shape)
        
        
        