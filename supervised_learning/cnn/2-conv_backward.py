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
    
    # print("dZ", dZ.shape)
    # print("A_prev", A_prev.shape)
    # print("W", W.shape)
    # print("b", b.shape)
    
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
    # print("db",db) # ok
    

    # dw = xp * dy
    # 0-padding juste sur les deux dernières dimensions de x
    xp = np.pad(A_prev, ((0,), (ph, ), (pw, ),(0,)), 'constant')
    # print("xp",xp.shape)
    dw = np.zeros(( kh, kw, c_prev, c_new))
    # Version sans vectorisation
    for n in range(m):       # On parcourt toutes les images
        for f in range(c_new):   # On parcourt tous les filtres
            for i in range(kh): # indices du résultat
                for j in range(kw):
                    for k in range(h_new): # indices du filtre
                        for l in range(w_new):
                            for c in range(c_prev): # profondeur
                                # print("xp and dz shape", xp.shape, dZ.shape)
                                dw[i,j, c, f] += xp[n, sh*i+k, sw*j+l,c] * dZ[n, k, l, f]

    # dx = dy_0 * w'
    # Valide seulement pour un stride = 1
    # 0-padding juste sur les deux dernières dimensions de dy = dout (N, F, H', W')
    doutp = np.pad(dZ, ((0,), (kh-1,), (kw-1, ), (0,)), 'constant')   # attention j'ai inveré kh et kw

    # 0-padding juste sur les deux dernières dimensions de dx
    dxp = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), 'constant')
    # print("dxp shape", dxp.shape, "A_prev shape", A_prev.shape)


    # filtre inversé dimension (F, C, HH, WW)
    w_ = np.zeros_like(W)
    for i in range(kh):
        for j in range(kw):
            w_[i,j, :, :] = W[kh-i-1,kw-j-1, :, :]
    
    # Version sans vectorisation
    for n in range(m):       # On parcourt toutes les images
        for f in range(c_new):   # On parcourt tous les filtres
            for i in range(h_prev): # indices du résultat
                for j in range(w_prev):
                    for k in range(kh): # indices du filtre
                        for l in range(kw):
                            for c in range(c_prev): # profondeur
                                dxp[n, i, j, c] += doutp[n, i+k, j+l, f] * w_[ k, l, c, f] 
    
    # Remove padding for dx
    #Remove padding for dx
    dx = dxp[:,:,ph:-ph,pw:-pw]
    # print("dx", dxp)
    # print("dx shape", dxp.shape)
    return dx, dw,db
        
        
        