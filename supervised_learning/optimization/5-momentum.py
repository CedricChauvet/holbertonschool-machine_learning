#!/usr/bin/env python3
"""
Optimization project
by Ced
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates a variable using the gradient descent with momentum
    optimization algorithm
    Returns: the updated variable and the new moment, respectively
    works for W and B
    """
    # Vm is the new momentum
    Vm = beta1 * v + (1 - beta1) * grad
    var = var - alpha * Vm

    return var, Vm
