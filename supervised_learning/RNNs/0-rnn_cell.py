#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import numpy as np


class RNNCell:

    """RNN cell class"""
    def __init__(self, i, h, o):
        self.Wh = np.random.normal(loc=0.0, scale=1.0, size=(h+i, h))
        self.Wy = np.random.normal(loc=0.0, scale=1.0, size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        # self.i = i
        # self.h = h
        # self.o = o

    def forward(self, h_prev, x_t):
        """Forward prop"""

        combined_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(combined_input, self.Wh) + self.bh)

        y = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y)
        return h_next, y

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)
