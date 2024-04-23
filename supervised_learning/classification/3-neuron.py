#!/usr/bin/env python3
"""
documentation création d'une classe neuro task 3: Neuron Cost

"""
import numpy as np


class Neuron:
    """ adding a cost function, and a logarithmic loss calcul """
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

    def set_A(self, p):
        """creation of a setter for A"""
        self.__A = p

    def forward_prop(self, X):
        """ take the Xnx imnut of a neuron
        the dot X with the weight and biases for
         forward propagation """
        forward_var = np.array(np.dot(X.T, self.__W.T) + self.__b)
        activation = np.array(1 / (1 + np.exp(-forward_var))).T
        self.set_A(activation)
        return self.__A

    def cost(self, Y, A):
        """ calcul of the cost function"""

        ni = Y.shape[1]
        """ need an intermediary part, calcul of the loss function"""
        loss = -(Y * np.log(A) + (1-Y) * np.log(1.0000001 - A))
        cost = (1 / ni) * np.sum(loss)
        return cost
