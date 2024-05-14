#!/usr/bin/env python3
"""
Optimization project
by Ced
"""
import matplotlib.pyplot as plt


def moving_average(data, beta):
    """
    calculates the weighted moving average
    of a data set:
    """

    v_avg = [0]
    bias = []
    result = [0]
    for i in range(1, len(data) + 1):
        bias.append(1 - (beta ** i))

        v_avg.append(beta * v_avg[-1] + (1 - beta) * (data[i-1]))

        result.append(v_avg[-1] / bias[-1])
    result.pop(0)

    return result
