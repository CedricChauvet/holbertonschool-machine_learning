#!/usr/bin/env python3
"""
Optimization project
by Ced
"""
import tensorflow as tf
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algorithm:
    the updated variable and the new moment, respectively
    see: https://www.youtube.com/watch?v=_e-LFe_
    igno&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=21
    """
    # S is the name of the RMSprop variable
    Sm = beta2 * s + (1 - beta2) * grad ** 2

    # adding epsilon to avoid nan numbers
    var = var - alpha * grad / (np.sqrt(Sm) + epsilon)

    # variable and the new moment
    return var, Sm
