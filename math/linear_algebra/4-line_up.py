#!/usr/bin/env python3

"""
Fonction contournant la methode
d'adition de vecteurs de numpy
"""


def add_arrays(arr1, arr2):
    sum = []
    if len(arr1) != len(arr2):
        return None
    else:
        for i in range(len(arr1)):
            sum.append(arr1[i] + arr2[i])
    return sum
