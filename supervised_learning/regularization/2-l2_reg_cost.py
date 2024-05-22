#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    task 2, got a model. using properies of model to get the cost
    """


    total_cost = cost  # Initialize with the base cost (without regularization)

    # Iterate through each layer in the model
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer:
            # If the layer has L2 regularization, add its contribution to the total cost
            reg_loss = layer.kernel_regularizer(layer.kernel)
            total_cost += reg_loss

    return total_cost



"""
    return cost + model.losses
"""