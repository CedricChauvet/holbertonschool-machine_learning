#!/usr/bin/env python3
"""
documentation création d'une classe neuro task 2: Neuron Forward Propagation
"""
import numpy as np


class Neuron:
    """putting private instance where
      @perperties are getter and setter added """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.nx = nx
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    @A.setter
    def A(self, p):
        self.__A = p

    def forward_prop(self, X):
        """ take the Xnx imnut of a neuron
        the dot X with the weight and biases for
         forward propagation """
        forward_var = np.array(np.dot(X.T, self.__W.T) + self.__b)
        activation = np.array(1 / (1 + np.exp(-forward_var))).T
        self.A = activation
        return self.__A
