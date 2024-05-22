#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """"
    Task 0 L2 Regularization cost

    calculates the cost of a neural network with L2 regularization:

    cost is the cost of the network without L2 regularization
    lambtha is the regularization parameter
    weights is a dictionary of the weights and biases (numpy.ndarrays)
    of the neural network
    L is the number of layers in the neural network
    m is the number of data points used

    Return: the cost of the network accounting for L2 regularization
    """

    cost_inter = 0

    for i in range(L):
        WL2 = np.sum(weights[f"W{i + 1}"] ** 2)
        cost_inter += lambtha / (2 * m) * WL2

    return cost + cost_inter
