#!/usr/bin/env python3
"""
probability project
"""


class Poisson:
    """
    Poisson distribution class
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Poisson distribution

        :param data: list of the data to be used to estimate the distribution
        :param lambtha: expected number of occurrences in a given time frame
        """
        self.data = data

        if data is not None:
            if isinstance(data, list):
                if len(data) >= 2:
                    self.lambtha = sum(data) / len(data)
                else:
                    raise ValueError("data must contain multiple values")
            else:
                raise TypeError("data must be a list")
        else:
            self.lambtha = float(lambtha)

        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")

    def pmf(self, k):
        """
        Probability mass function

        :param k: number of occurrences
        :return: PMF value for k occurrences
        """
        e = 2.7182818285
        k = int(k)
        if k < 0:
            return 0

        return pow(self.lambtha, k) * pow(e, -self.lambtha) / factorielle(k)

    def cdf(self, k):
        cdf = 0
        k = int(k)
        for i in range(k):
            pmf_i = self.pmf(i)
            cdf += pmf_i
        return cdf
    
def factorielle(n):
    """
    alog factoriel
    """

    if n == 0:
        return 1
    else:
        F = 1
    for k in range(2, n + 1):
        F = F * k

    return F
