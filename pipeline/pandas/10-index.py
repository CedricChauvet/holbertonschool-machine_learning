#!/usr/bin/env python3
"""
Pandas project
By Ced
"""


def index(df):
    """
    set the index of the dataframe
    """
    df = df.set_index('Timestamp')
    return df
