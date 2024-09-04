#!/usr/bin/env python3
"""
Rnns project
use the rnn class of the rnn module task0
"""
import numpy as np


class LSTMCell():
    """
    LSTMCell class
    """

    def __init__(self, i, h, o):
        self.Wf = np.random.normal(size=(h+i, h))
        self.Wu = np.random.normal(size=(h+i, h))
        self.Wc = np.random.normal(size=(h+i, h))
        self.Wo = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1,h))
        self.bu = np.zeros((1,h))
        self.bc = np.zeros((1,h))
        self.bo = np.zeros((1,h))
        self.by = np.zeros((1,o))

        def forward(self, h_prev, c_prev, x_t):
            """
            forward, from left to right
            """
            h_next = None
            c_next = None
            y = None
            return h_next, c_next, y


