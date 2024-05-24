#!/usr/bin/env python3
"""
Regularization project
by Ced
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    task 7 early stopping, end of the project

    cost is the current validation cost of the neural network
    opt_cost is the lowest recorded validation cost of the neural network
    threshold is the threshold used for early stopping
    patience is the patience count used for early stopping
    count is the count of how long the threshold has not been met
    """
    if(opt_cost - cost) > threshold:
        return (False, 0)

    elif(opt_cost - cost) <= threshold:
        if count < patience - 1:
            return (False, count + 1)
        if count == patience - 1:
            return(True, patience)
