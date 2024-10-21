#!/usr/bin/env python3
"""
Project Q learning
by Ced
"""
import gymnasium as gym
import numpy as np
load_frozen_lake = __import__('0-load_env').load_frozen_lake


def q_init(env):
    """
    Initializes the Q-table
    4 actions in the Q-table up-down-left-right
    16 states in the Q-table lors de table 4*4
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    return Q
    