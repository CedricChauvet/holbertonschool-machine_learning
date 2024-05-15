#!/usr/bin/env python3
"""
Optimization project
by Ced
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Write the function def update_variables_Adam(alpha, beta1,
    beta2, epsilon, var, grad, v, s, t): that updates a variable in
    place using the Adam optimization algorithm:

    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    v is the previous first moment of var
    s is the previous second moment of var
    t is the time step used for bias correction

    Returns: the updated variable, the new first moment, and the
    new second moment, respectively
    """

    # Vm is the new momentum
    Vm = beta1 * v + (1 - beta1) * grad
    # S is the name of the RMSprop variable
    Sm = beta2 * s + (1 - beta2) * grad ** 2

    # calculates the weighted moving average, see task 4
    biasV = (1 - (beta1 ** t))
    biasS = (1 - (beta2 ** t))

    Vmcorr = Vm / biasV
    Smcorr = Sm / biasS  # adding epsilon to avoid nan numbers

    var = var - alpha * Vmcorr / (np.sqrt(Smcorr) + epsilon)

    return var, Vm, Sm
