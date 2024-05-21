#!/usr/bin/env python3
"""
project Error Analysis
By Ced
"""
import numpy as np


def sensitivity(confusion):
    """
    return number of TP / (TP + FN ), accross the line
    """

    classes = confusion.shape[1]
    sensitivity = np.zeros(classes)
    for i in range(classes):
        TP = confusion[i, i]
        set = np . sum(confusion[i, :])
        sensitivity[i] = TP / set

    return sensitivity
