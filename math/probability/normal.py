#!/usr/bin/env python3
"""
probability project
"""


class Normal():
    """
    class for a gaussian distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):

        if data is None:
            self.mean = mean
            self.stddev = stddev
            if stddev < 0 or not isinstance(stddev, float):
                raise ValueError("stddev must be a positive value")

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data)/len(data)
            variance = 0
            for val in data:
                variance += (val - self.mean) ** 2 / len(data)

            self.stddev = pow(variance, 1/2)
