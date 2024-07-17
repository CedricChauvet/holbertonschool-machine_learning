#!/usr/bin/env python3
"""
probability project
"""


class Poisson():
    """
    Task 0  constructor
    """

    def __init__(self, data=None, lambtha=1.):
        """
        lambtha is the expected number of occurences in a given time frame
        data is a list of the data to be used to estimate the distribution
        """
        self.lambtha = lambtha
        self.data = data

        if type(self.data) is list:

            if len(data) >= 2:
                s = 0
                for i in data:
                    s += i
                mean = s / len(data)
                self.lambtha = mean
            else:
                raise ValueError("data must contain multiple values")

        elif self.data is None:
            pass

        else:
            raise TypeError("data must be a list")

        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
