#!/usr/bin/env python3

"""
Fonction contournant la methode
d'adition de matrice element par element
de vecteurs de numpy
"""


def add_matrices2D(mat1, mat2):
    """
    fonction de la task 5
    """
    row = []
    sum = []
    if len(mat1) == 0:
        return None
    else:
        for i in range(0, len(mat1)):
            for j in range(0, len(mat1[0])):
                row.append(mat1[i][j] + mat2[i][j])
            sum.append(row)
            row = []
    return sum
