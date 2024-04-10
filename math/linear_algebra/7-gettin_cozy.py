#!/usr/bin/env python3

"""
Task 7, algebra project
"""


def cat_arrays(arr1, arr2):
    """
    Function that concatenates two arrays
    """

    arr = []
    for i in range(len(arr1)):
        arr.append(arr1[i])

    for i in range(len(arr2)):
        arr.append(arr2[i])

    return arr


def cat_matrices2D(mat1, mat2, axis=0):
    """
    fonction that two 2D matrices along a specific axis
    numpy not implemented!
    """
    matc = []

    if axis == 0:
        """ concatenate along the row, add a row """

        a = len(mat1[0])
        """ verify a matrix suit the over, number of columns """
        for k in range(len(mat1)):
            if len(mat1[k]) != a:
                return None
        for m in range(len(mat2)):
            if len(mat2[m]) != a:
                return None

        for j in range(len(mat1)):
            matc.append(mat1[j])
        for i in range(len(mat2)):
            matc.append(mat2[i])
        return matc

    if axis == 1:
        """ concatenate along the columns, add a column"""
        matc = []

        if len(mat1) != len(mat2):
            return None

        for i in range(len(mat1)):
            matc.append(cat_arrays(mat1[i], mat2[i]))

        return matc

    else:
        """ en cas d'erreurs """
        return None
