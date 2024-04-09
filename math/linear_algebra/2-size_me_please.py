#!/usr/bin/env python3

"""
Fonction contournant la methode shape de numpy
"""


def matrix_shape(matrix):
    dims = []

    while type(matrix) == list:
        dims.append(len(matrix))
        matrix = matrix[0]
    return dims
