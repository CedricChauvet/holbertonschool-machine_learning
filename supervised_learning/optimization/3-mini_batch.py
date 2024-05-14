#!/usr/bin/env python3
"""
Optimization project
by Ced
"""
import numpy as np
import tensorflow as tf
import random
import os

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    creates mini-batches to be used for training
    a neural network using mini-batch gradient descent:
    """

    # init and shuffle the data
    X, Y = shuffle_data(X, Y)
    b = 0
    batch = []

    # Create mini-batches
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        batch.append((X_batch, Y_batch))

    return batch
