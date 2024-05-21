#!/usr/bin/env python3
"""
project Error Analysis
By Ced
"""
import numpy as np


def precision(confusion):
    """
     calculates the precision for each class
     in a confusion matrix:

    confusion is a confusion numpy.ndarray of
    shape (classes, classes) where row indices
    represent the correct labels and column indices
    represent the predicted labels
    classes is the number of classes

    Returns: a numpy.ndarray of shape (classes,)
    containing the precision of each class
    """

    classes = confusion.shape[1]
    precision = np.zeros(classes)
    for i in range(classes):
        TP = confusion[i, i]
        set = np . sum(confusion[:, i])
        precision[i] = TP / set

    return precision
