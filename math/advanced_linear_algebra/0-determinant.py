#!/usr/bin/env python3
"""
Project linear algebra
"""


def determinant(matrix):
    """
    make the determinant of a matrix
    """
    if not all(isinstance(lst,list)  for lst in matrix):

        raise TypeError("matrix must be a list of lists")
    
    first_len = len(matrix[0])
    
    if not all(len(lst) == first_len for lst in matrix) and len(matrix) == first_len:
        raise ValueError("matrix must be a square matrix")

    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return  (matrix[0][0]* matrix[1][1]) - (matrix[1][0]* matrix[0][1])  
    


    det = 0
    for i in range(len(matrix)):
        det += ((-1) ** i) *  matrix[0][i] * determinant(get_matrix_minor(matrix, i))
    return det

   
def get_matrix_minor(matrix, j):
    """
    very interesting solution
    """
    return [row[:j] + row[j+1:] for row in ( matrix[1:])]









def sous_matrice(matrix, i):
    """
    ne fonctionne mas
    """
    det = matrix
    del det[i]
    det = transpose(det)
    del det[i]
    det = transpose(det)
    return det

def transpose(list_of_lists):
    """
    used to trasnpose
    """
    return list(map(list, zip(*list_of_lists)))