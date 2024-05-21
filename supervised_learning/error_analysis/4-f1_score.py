#!/usr/bin/env python3
"""
project Error Analysis
By Ced
"""
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    calculates the F1 score of a confusion matrix:
    """
    classes = confusion.shape[0]
    total = np.sum(confusion)

    F1 = 2 * (precision(confusion) * sensitivity(confusion))\
        / (precision(confusion) + sensitivity(confusion))

    return F1
