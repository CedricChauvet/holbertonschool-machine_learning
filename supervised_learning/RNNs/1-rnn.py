#!/usr/bin/env python3
"""
Rnns project
use the rnn class of the rnn module task0
"""
import numpy as np
def rnn(rnn_cell, X, h_0):
    """
    forward prop for a simple RNN
    X is the data to be used
    h_0 is the initial hidden state
    """

    t, m, i = X.shape
    _, h = h_0.shape
    
    # shape of H
    H = np.zeros((t+1, m, h))
    H[0] = h_0
    
    # Initialize Y with the correct shape
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))  # Assuming rnn_cell.Wy exists

    # make n time steps for the rnn
    for j in range(t):
        H[j+1], Y[j] = rnn_cell.forward(H[j], X[j])
    
    print("Y shape:", Y.shape)
    print("Y[0] shape:", Y[0].shape)
    
    return H, Y