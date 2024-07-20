#!/usr/bin/env python3
"""
probability project
"""


class Exponential:
    """
    Poisson distribution class
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the exponantial distribution

        :param data: list of the data to be used to estimate the distribution
        :param lambtha: expected number of occurrences in a given time frame
        """
        self.data = data

        if data is not None:
            if isinstance(data, list):
                if len(data) >= 2:
                    self.lambtha = len(data) / sum(data)
                else:
                    raise ValueError("data must contain multiple values")
            else:
                raise TypeError("data must be a list")
        else:
            self.lambtha = float(lambtha)

        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")

    def pdf(self, x):
        """
        return probability density function
        """
        e = 2.7182818285
        if x < 0:
            return 0
        else:
            return self.lambtha * pow(e, -x * self.lambtha)

    def cdf(self, k):
        """
        cdf :  fonction de répartition cumulative
        :param k: number of occurrences
        :return: cdf value for k occurrences
        """
        e = 2.7182818285
        if k < 0:
            return 0
        
        else:
            return 1 - pow(e, - self.lambtha * k)