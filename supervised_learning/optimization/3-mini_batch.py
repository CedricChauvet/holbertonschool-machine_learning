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

    # init and shuffle the
    X_sh, Y_sh = shuffle_data(X, Y)
    batch = []
    b = 0
    # use of while is a good option
    while b < X_sh.shape[0] - batch_size:
        batch.append((X_sh[b:b+batch_size], Y_sh[b:b+batch_size]))
        b += batch_size

    batch.append((X_sh[b:], Y_sh[b:]))
    # print("dernier batch",len(batch[-1][0]))

    return batch
