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

    def pmf(self, k):
        """
        Fonctions de masse de probabilité
        """
        e = 2.7182818285
        return pow(self.lambtha, k) * pow(e, -self.lambtha) / factorielle(k)


def factorielle(n):
    """
    alog factoriel
    """

    if n < 0:
        n = 0
    if n == 0:
        return 1
    else:
        F = 1
    for k in range(2, n + 1):
        F = F * k

    return F

