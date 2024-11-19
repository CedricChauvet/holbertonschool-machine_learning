#!/usr/bin/env python3
"""
Pandas project
By Ced
"""
import pandas as p


def slice(df):
    """
    slice a dataframe
    """
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']][::60]
    return df.tail()
