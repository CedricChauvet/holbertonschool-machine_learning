#!/usr/bin/env python3
"""
Regularization project
by Ced
"""


def l2_reg_cost(cost, model):
    """
    task 2, got a model. using properies of model to get the cost
    """

    return cost + model.losses
