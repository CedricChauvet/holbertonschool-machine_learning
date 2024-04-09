#!/usr/bin/env python3

"""
Fonction contournant la methode
transpose de numpy pour une matrice de dimension 2
"""


def matrix_transpose(matrix):
    """ fonction pour remplir la task 2
    """

    row = []
    mat = []
    for i in range(0, len(matrix[0])):
        for j in range(0, len(matrix)):
            row.append(matrix[j][i])
        mat.append(row)
        row = []
    return mat
