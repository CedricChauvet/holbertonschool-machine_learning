#!/usr/bin/env python3
"""
Task 8, algebra project : Ridin’ Bareback
"""


def select_array(matrice, n, axis):
    """ choose a row or a column in the matrix"""

    array = []
    """vecteur vertical"""
    if axis == 0:
        for i in matrice:
            array.append(i[n])

    """vecteur horizontal"""
    if axis == 1:
        array = matrice[n]
    return array


def product_array(array1, array2):
    """ product of two identical arrays"""

    sum = 0
    if len(array1) != len(array2):
        return None

    for i in range(len(array1)):
        sum += array1[i] * array2[i]
    return sum


def mat_mul(mat1, mat2):
    """ writing a function that permos matrix multiplication, without numpy """

    a1 = []
    a2 = []
    product_matrix_elmt = []
    product_matrix = []
    mat_result = []

    if len(mat1[0]) != len(mat2):
        return None

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            a1 = select_array(mat1, i, 1)
            a2 = select_array(mat2, j, 0)
            intproduct = product_array(a1, a2)
            product_matrix.append(intproduct)
        mat_result.append(product_matrix)
        product_matrix = []
    return mat_result
