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
            if stddev < 0 or not isinstance(stddev, (float, int)):
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

    def z_score(self, x):
        """
        return standard normal deviation
        """

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        return to  normal deviation
        """

        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        return probability density for the Gaussian distribution
        """
        pi = 3.1415926536
        e = 2.7182818285

        return 1 / pow(2 * pi * (self.stddev ** 2), 1 / 2) *\
            pow(e, -(x - self.mean) ** 2 / (2 * (self.stddev ** 2)))

    def cdf(self, x):
        """
        return cumulative distribution function
        """
        return 1 / 2 * (1 + self.erf((x - self.mean)
                        / (self.stddev * pow(2, 1/2))))

    def erf(self, x):
        """
        erf is a mathematical expression
        """
        pi = 3.1415926536
        return 2 / pow(pi, 1/2) * (x - (pow(x, 3) / 3)
                                   + (pow(x, 5) / 10) - (pow(x, 7) / 42)
                                   + (pow(x, 9) / 216))
