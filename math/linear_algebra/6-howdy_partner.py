#!/usr/bin/env python3

"""
task 5, algebra project
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
