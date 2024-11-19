#!/usr/bin/env python3
"""
Pandas project
By Ced
"""


def array(df):
    """
    selecting the last 10 rows of the 2 columns High and Close
    and convert them into an array numpy
    """
    df = df[['High', 'Close']]
    df = df.tail(10)
    df_array = df.to_numpy()
    return df_array
