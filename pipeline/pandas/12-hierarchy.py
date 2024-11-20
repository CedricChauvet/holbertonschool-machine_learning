#!/usr/bin/env python3
"""
Pandas project
By Ced
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    create a hierarchy for the data
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[(df2.index <= 1417417980) & (df2.index >= 1417411980)]
    df1 = df1.loc[(df1.index <= 1417417980) & (df1.index >= 1417411980)]
    df = pd.concat([df2, df1],  keys=['bitstamp', 'coinbase'])
    df = df.sort_index(level='Timestamp')
    df = df.swaplevel(0, 1)
    return df
