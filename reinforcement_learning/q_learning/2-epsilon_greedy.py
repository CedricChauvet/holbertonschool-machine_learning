#!/usr/bin/env python3
"""
Project Q learning
by Ced
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action
    """
    Q = Q[state]
    p = np.random.uniform(0, 1) - epsilon
    if p > 0:
        """
        exploitation
        """
        action = np.argmax(Q)
    else:
        """
        or exploration
        """
        action = np.random.randint(0, len(Q))
    return action


