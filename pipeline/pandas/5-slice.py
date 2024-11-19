#!/usr/bin/env python3
"""
Pandas project
By Ced
"""


def slice(df):
    """
    slice a dataframe
    """
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']][::60]
    return df.tail()
