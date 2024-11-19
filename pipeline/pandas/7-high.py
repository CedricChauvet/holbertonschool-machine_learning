#!/usr/bin/env python3
"""
Pandas project
By Ced
"""


def high(df):
    """
    Sorts it by the High price in descending order
    return the sorted data frame
    """

    df = df.sort_values(by=['High'], ascending=False)
    return df
