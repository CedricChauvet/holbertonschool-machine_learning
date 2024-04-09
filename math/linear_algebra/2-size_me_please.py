#!/usr/bin/env python3

"""
Fonction contournant la methode shape de numpy
"""


def matrix_shape(matrix):
    """
    prend une matrice en parametre et retourne sa forme
    """

    dims = []
    while type(matrix) is list:
        dims.append(len(matrix))
        matrix = matrix[0]
    return dims
