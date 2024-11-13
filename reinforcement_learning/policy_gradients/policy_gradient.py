#!/usr/bin/env python3
"""
This project is about policy gradient
By Ced
"""
import numpy as np


def policy(matrix, weight):
    """
    Function that computes policy with a weight of a matrix
    weight: matrix of random weight, my policy
    matrix: state or observation of the environment
    returns softamx policy
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z)
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """
    find the gradient J of the policy
    """
    # takes useful variables
    n_states, n_actions = weight.shape

    # Initialiser le gradient avec la bonne forme
    gradient = np.zeros((n_states, n_actions))

    policy_value = policy(state, weight)
    #print("policy_value: ", policy_value[0])
    # stochastic policy based on previous task
    if np.random.random() > policy_value[0]:
        action = 1
    else:
        action = 0

    # Calculer le gradient pour chaque action
    for a in range(n_actions):
        if a == action:
            gradient[:, a] = state * (1 - policy_value[a])
        else:
            gradient[:, a] = -state * policy_value[a]

    return action, gradient
