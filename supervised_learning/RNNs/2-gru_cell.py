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
        h_next = None
        y = None
        return h_next, y 
        