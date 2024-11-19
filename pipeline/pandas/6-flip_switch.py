#!/usr/bin/env python3
"""
Pandas project
By Ced
"""


def flip_switch(df):
    """
    flip the dataframe
    then transpose it
    """

    df = df[::-1]
    df = df.transpose()

    return df
