#!/usr/bin/env python3*
"""
task 10, project calculus: Derive happiness in oneself from a good day's work
"""


def poly_derivative(poly):
    """ This is a documentation """
    list2 = []
    if type(poly) is not list:
        return None
    for i in poly:
        if type(i) is not int:
            return None

    if len(poly) == 1:
        return [0]
    for i in range(0, len(poly) - 1):

        list2.append(poly[i+1]*(i+1))
    return list2
