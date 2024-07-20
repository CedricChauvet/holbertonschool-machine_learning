#!/usr/bin/env python3
"""
probability project
"""
import numpy as np

class Normal():
    """
    class for a gaussian distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):

        if data is None:
            self.mean = mean
            self.stddev = stddev
            if stddev < 0:
                raise ValueError ("stddev must be a positive value")
        
        else:
            if not isinstance(data, list):
                raise TypeError ("data must be a list")
            
            if len(data) < 2:
                raise ValueError ("data must contain multiple values")  
            


            data_sorted = sorted(data)
            i = len(data) // 2
            self.mean = data_sorted[i-2]
            self.stddev =  1
