#!/usr/bin/env python3
"""
Task 9, project calculus: Our life is the sum total
of all the decisions we make every day,
and those decisions are determined by our priorities
"""


def summation_i_squared(n):
    """this is a documentation"""
    
    if n == None:
        return None
    if type(n) is not int:
        return None
    if n <= 0:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)


