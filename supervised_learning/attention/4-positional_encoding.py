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

    # Calcul de chaque position et dimension
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            # Calcul des valeurs de sinus et cosinus
            angle = pos / np.power(10000, (2 * (i // 2)) / dm)
            PE[pos, i] = np.sin(angle)
            if i + 1 < dm:
                PE[pos, i + 1] = np.cos(angle)

    return PE
