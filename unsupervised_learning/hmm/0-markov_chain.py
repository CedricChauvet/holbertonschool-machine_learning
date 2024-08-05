#!/usr/bin/env python3
"""
Project Hiden Markov Model
By Ced+
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    beginning with task 0
    """
    return s @  np.linalg.matrix_power(P, t)
