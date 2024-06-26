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
    # print("A_prev", A_prev.shape)
    # print("W", W.shape)
    # print("b", b.shape)
    
    (m, h_new, w_new, c_new) = dZ.shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    (sh, sw) = stride
    print("stride", sh,sw)
    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        ph = int((h_prev * (sh - 1) + kh - sh) / 2)
        pw = int((w_prev * (sw - 1) + kw - sw) / 2)

    dw = np.zeros(( kh, kw, c_prev, c_new))
    dxp = np.zeros((m, h_prev + 2 *ph, w_prev + 2 * pw , c_prev))

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
    dZp = np.pad(dZ, ((0,), (kw-1,), (kh-1, ), (0,)), 'constant')   # attention j'ai inveré kh et kw
    print("dzp", dZp.shape)
    # 0-padding juste sur les deux dernières dimensions de dx
    dxp = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), 'constant')
    # print("dxp shape", dxp.shape, "A_prev shape", A_prev.shape)


    # filtre inversé dimension (F, C, HH, WW)
    W_ = np.zeros_like(W)
    for i in range(kh):
        for j in range(kw):
           W_[i,j, :, :] = W[kh-i-1,kw-j-1, :, :]
    
    
    # Version sans vectorisation
    for n in range(m):       # Parcours des images
        for f in range(c_new):   # Parcours des filtres
            for i in range(h_prev + 2 * ph): # Parcours des indices de hauteur de l'entrée
                for j in range(w_prev + 2 * pw): # Parcours des indices de largeur de l'entrée
                    for k in range(kh): # Parcours des indices de hauteur du filtre
                        for l in range(kw): # Parcours des indices de largeur du filtre
                            for c in range(c_prev): # Parcours des canaux
                                #if (i - k) % sh == 0 and (j - l) % sw == 0:
                                #    ii = (i - k) // sh
                                #    jj = (j - l) // sw
                                #    if 0 <= ii < h_new and 0 <= jj < w_new:
                                    dxp[n, i, j, c] += dZp[n, sh*i +k, sw*j +l, f] * W_[k, l, c, f ]
  
    # Remove padding for dx
    #Remove padding for dx
    if padding == "same":
        dx = dxp[:,ph:-ph,pw:-pw,:]
        # print("dx", dx.shape)
    if padding == "valid":
        dx = dxp
    # print("dxp shape", dxp.shape)
   
    return dx, dw,db
        
        
        