#!/usr/bin/env python3
"""
Rnns project
use the rnn class of the rnn module task0
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    
    # rnn_cells is a list of RNNCell instances of length l 
    # that will be used for the forward propagation

    # X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    # t is the maximum number of time steps
    # m is the batch size
    # i is the dimensionality of the data
    t, m, i = X.shape

    # h_0 is the initial hidden state, given as a numpy.ndarray of shape (l, m, h)
    # h is the dimensionality of the hidden state
    l, m, h = h_0.shape

    # layer 1, 2, 3, tune with boucle for after

    H1, Y1 =  rnn(rnn_cells[0], X, h_0[0])
    H2, Y2 =  rnn(rnn_cells[1], Y1, h_0[1])
    H3, Y3 =  rnn(rnn_cells[2], Y2, h_0[2])
    
    print("H1 shape:", H1.shape)
    print("H2 shape:", H2.shape)
    print("H3 shape:", H3.shape)
    return H3, Y3



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

    return H, Y