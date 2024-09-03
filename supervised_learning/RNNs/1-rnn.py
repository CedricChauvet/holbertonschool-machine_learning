#!/usr/bin/env python3
"""
Rnns project
use the rnn class of the rnn module task0
"""
import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell


def rnn(rnn_cell, X, h_0):
    """
    forward prop for a simple RNN
    X is the data to be used
    h_0 is the initial hidden state
    """

   
    t, m, i = X.shape
    _, h = h_0.shape
    Hs = np.zeros((t, m, h))
    for j in range(t):
        if j == 0:
            Hs[j] = h_0
        
        if j != 0:
            Hs[j], _ = rnn_cell.forward(Hs[j-1], X[j] )

        

    return Hs, None