#!/usr/bin/env python3
"""
project Error Analysis
By Ced
ONLY WORKS ON ONEHOT arrays

"""
import numpy as np
# Applying the function to training set labels and testing set labels 
from keras.utils import to_categorical 


def one_hot_encode(train_labels, test_labels):
    """
    convert to onehot matrix
    """
    train_labels = to_categorical(train_labels, dtype ="uint8") 
    test_labels = to_categorical(test_labels, dtype ="uint8")


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix:

    labels is a one-hot numpy.ndarray
    of shape (m, classes) containing
    the correct labels for each data point
    m is the number of data points
    classes is the number of classes
    logits is a one-hot numpy.ndarray
    of shape (m, classes) containing the predicted labels

    Returns: a confusion numpy.ndarray of shape (classes, classes)
    """




    m = labels.shape[0]
    classes = labels.shape[1]
    mat_confusion = np.zeros((classes, classes))

    for i in range(m):

        labx = np.argwhere(labels[i] == 1)
        logx = np.argwhere(logits[i] == 1)

        mat_confusion[labx, logx] += 1

    return mat_confusion


# to plot with colors and matplotlib