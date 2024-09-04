#!/usr/bin/env python3
"""
Rnns project
use the rnn class of the rnn module task0
"""
import numpy as np


class LSTMCell():
    """
    LSTMCell class
    https://penseeartificielle.fr/comprendre-lstm-gru-fonctionnement-schema/
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
        

        z = np.concatenate((h_prev, x_t), axis=1)
        z1 = self.sigmoid(np.dot(z, self.Wf) + self.bf)
        z2 = (self.sigmoid(np.dot(z,self.Wu)+self.bu)) * (np.tanh(np.dot(z,self.Wc)+self.bc))
        z3 = self.sigmoid(np.dot(z,self.Wo)+self.bo)
        c_next = c_prev * z1 + z2
        h_next = np.tanh(c_next) * z3

        y = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y)

        return h_next, c_next, y


    def sigmoid(self, x):
        """
        compute sigmoid values for each sets of scores in x
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)    

