#!/usr/bin/env python3
"""
project Error Analysis
By Ced
"""
import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a
    confusion matrix:

    confusion is a confusion numpy.ndarray of shape
    (classes, classes) where row indices represent the
    correct labels and column indices represent the predicted labels
    classes is the number of classes

    Returns: a numpy.ndarray of shape (classes,) containing the specificity
    of each class

    specificity = TN / (TN + FP)

    """

    classes = confusion.shape[0]
    specificity = np.zeros(classes)
    total = np.sum(confusion)

    for i in range(classes):
        inter1 = np.sum(confusion[i, :])
        inter2 = np.sum(confusion[:, i])
        TP = confusion[i, i]
        TN = total - inter1 - inter2 + TP
        FP = np.sum(confusion[:, i]) - confusion[i, i]

        specificity[i] = TN / (TN + FP)

    return specificity
