#!/usr/bin/env python3
"""
First task with a new class: NeuralNetwork
Write a class NeuralNetwork that defines a neural network with one
hidden layer performing binary classification

"""
import numpy as np


class NeuralNetwork:
    """
    class Neural network
    """
    def __init__(self, nx, nodes):

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nodes = nodes

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__A1 = np.array(np.dot(X.T, self.__W1.T) + self.__b1.T)
        self.__A2 = np.array(np.dot(self.__A1, self.__W2.T) + self.__b2)

        self.__A1 = np.array(1 / (1 + np.exp(-self.__A1))).T
        self.__A2 = np.array(1 / (1 + np.exp(-self.__A2))).T
        return self.__A1, self.__A2