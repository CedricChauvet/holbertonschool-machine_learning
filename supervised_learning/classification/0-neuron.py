#!/usr/bin/env python3
"""
documentation création d'une classe neuro
"""
import numpy as np


class Neuron:
    """ construcion brique par brique, here build the constructor only"""
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.nx = nx
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
