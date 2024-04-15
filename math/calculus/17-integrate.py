#!/usr/bin/env python3
"""
Task 17 project calculus: Integrate
"""


def poly_integral(poly, C=0):
    """ This is a documentation"""
    
    poly_int=[C]
    
    
    if poly= None:
        return poly_int
    for i in poly:
        if type(i) not in (float, int):
            return None
 
    for i in range(len(poly)):
        
        if poly[i]/(i + 1) % 1 <= 0.0:
            poly_int.append(int(poly[i]/(i + 1)))
        
        elif isinstance( poly[i]/(i + 1), float):
            poly_int.append(float(poly[i]/(i + 1)))
    return poly_int
