#!/usr/bin/env python3
"""
Rnns project
use the rnn class of the rnn module task0
"""
import numpy as np


class GRUCell():
    def __init__(self,i, h, o):
        self.i = i
        self.h = h
        self.o = o

        self.Wz= np.random.normal(size=(h+i, h))
        self.bz = np.zeros((1,h))
        self.Wr= np.random.normal(size=(h+i, h))
        self.br = np.zeros((1,h))
        self.Wh= np.random.normal(size=(h+i, h))
        self.bh = np.zeros((1,h))
        self.Wy= np.random.normal(size=(h,o ))
        self.by = np.zeros((1,o))
        
    def forward(self, h_prev, x_t):
        """
        toute la diffuculté est d'ecrire efficacement les equations de la cellule GRU
        """
        z = np.concatenate((h_prev, x_t), axis=1)
        out_Reset = self.sigmoid(np.dot(z, self.Wr) + self.br) * h_prev
        out_Update = self.sigmoid(np.dot(z, self.Wz) + self.bz)
        out_h= np.tanh(np.dot(np.concatenate((out_Reset, x_t), axis=1),self.Wh) + self.bh)
        
        # print("out_Reset", out_Reset.shape)
        # print("out_Update", out_Update.shape)
        # print("out_h", out_h.shape)

        # h_i0 et h_next
        h_i0 = h_prev * (1 - out_Update)
        h_next = h_i0 + (out_Update * out_h)
        y = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y)

        # print("out_Reset", out_Reset.shape)

        return h_next, y 
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)
