#!/usr/bin/env python3
"""
Project linear algebra
"""

def minor(matrix):
    """
    get minor
    """

    if not all(isinstance(lst, list) for lst in matrix):
        raise TypeError("matrix must be a list of lists")

    first_len = len(matrix[0])
    if not all(len(lst) == first_len for lst in matrix)\
            or len(matrix) != first_len or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return 1

    sol2D = []
    for i in range(len(matrix)):
        sol = []
        for j in range(len(matrix)):
            minor_mat = get_matrix_minor(matrix, i, j)
            det_minor = determinant(minor_mat)
            sol.append(det_minor)
        sol2D.append(sol)
    return sol2D


def determinant(matrix):
    """
    make the determinant of a matrix
    """

    if matrix == [[]]:
        return 1

    if not all(isinstance(lst, list) for lst in matrix):
        raise TypeError("matrix must be a list of lists")

    first_len = len(matrix[0])
    if not all(len(lst) == first_len for lst in matrix)\
            or len(matrix) != first_len:
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1])

    det = 0
    for i in range(len(matrix)):
        det += ((-1) ** i) * matrix[0][i]\
            * determinant(get_matrix_minor(matrix, 0, i))
    return det


def get_matrix_minor(matrix, i, j):
    """
    very interesting solution
    """
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
