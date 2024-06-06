#!/usr/bin/env python3
"""
Convolution and pooling project
by Ced
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    to document
    """
    (m, h_new, w_new, c_new) = dA.shape
    (m, h_prev , w_prev, c) = A_prev.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride
    """
    print("dA shape", dA.shape)
    print("A_prev shape", A_prev.shape)
    print("kernel ", kernel_shape)
    print("stride ", stride)
    """
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    
                    if mode == "max":
                        tmp = A_prev[i, h*sh:kh+(h*sh),
                                        w*sw:kw+(w*sw), f]
                        mask = (tmp == np.max(tmp))
                        dA_prev[i,
                            h*(sh):(h*(sh))+kh,
                            w*(sw):(w*(sw))+kw,
                            f] += dA[i, h, w, f] * mask
                    
                    if mode == "avg":
                        dA_prev[i, h*(sh):(h*(sh))+kh, w*(sw):(w*(sw))+kw, f] += (dA[i, h, w, f])/kh/kw
    
    return dA_prev
    