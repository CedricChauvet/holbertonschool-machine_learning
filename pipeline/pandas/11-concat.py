#!/usr/bin/env python3
"""
Pandas project
By Ced
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    concatenate 2 dataframes
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[df2.index <= 1417411920]
    df = pd.concat([df2, df1],  keys=['bitstamp', 'coinbase'])
    return df
