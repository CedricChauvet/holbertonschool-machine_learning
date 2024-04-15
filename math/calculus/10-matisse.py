#!/usr/bin/env python3*
"""
task 10, project calculus: Derive happiness in oneself from a good day's work
"""


def poly_derivative(poly):
    """ This is a documentation """
    list2 = []
    for i in range (len(poly)):
        if type(poly[i]) is not int:
            return None

    if len(poly) == 1:
        return [0]
    for i in range(0, len(poly) - 1):

        list2.append(poly[i+1]*(i+1))
    return list2
