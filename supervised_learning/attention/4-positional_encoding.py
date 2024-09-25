#!/usr/bin/env python3
"""
Attention project
By Ced
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer
    param max_seq_len: integer representing the maximum sequence length
    param dm: integer representing the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing
    the positional encoding vectors
    """
    PE = np.zeros((max_seq_len, dm))
    for i in range(max_seq_len):
        for j in range(0, dm, 2):
            PE[i, j] = np.sin(i / (10000 ** ((2 * j) / dm)))
            PE[i, j + 1] = np.cos(i / (10000 ** ((2 * j) / dm)))

    return PE