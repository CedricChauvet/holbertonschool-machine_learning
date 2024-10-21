#!/usr/bin/env python3
"""
Project Q learning
by Ced
"""
import numpy as np



def q_init(env):
    """
    Initializes the Q-table
    4 actions in the Q-table up-down-left-right
    16 states in the Q-table lors de table 4*4
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    return Q
    