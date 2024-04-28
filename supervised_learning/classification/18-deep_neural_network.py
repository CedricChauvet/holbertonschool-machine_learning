#!/usr/bin/env python3
"""
new class: DeepNeuralNetwork, task  18:  DeepNeuralNetwork
task 18: DeepNeuralNetwork Forward Propagation
"""
import numpy as np


class DeepNeuralNetwork:
    """ define a new class with private attributes"""
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.layers = layers
        self.__weights = dict()
        self.__L = len(layers)
        self.__cache = dict()

        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights['W1'] = np.random.randn(
                        layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W{}'.format(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights['b{}'.format(i + 1)] = np.zeros((layers[i], 1))

        # He et al. method:
        # w=np.random.randn(layer_size[l],layer_size[l-1])*np.sqrt(2/layer_size[l-1])

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """ forward propagation in a DNN, be careful of Matrix dimensions"""
        for i in range(self.__L + 1):
            if i == 0:
                self.__cache['A0'] = X
            else:
                z = np.dot(self.__weights['W{}'.format(
                    i)], self.__cache['A{}'.format(i - 1)]
                    ) + self.__weights['b{}'.format(i)]
                sigmoid_z = 1 / (1 + np.exp(-z))
                self.__cache['A{}'.format(i)] = sigmoid_z
        # retourne le dernier Layer, puis l'ensemble des
        # données dans cache, toutes les activations
        return self.__cache['A{}'.format(self.__L)], self.__cache
