#!/usr/bin/env python3


def matrix_shape(matrix):
    dims = []
    print("matrix ", matrix)
    while type(matrix) == list:
        dims.append(len(matrix))
        matrix = matrix[0]
    return dims
