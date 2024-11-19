#!/usr/bin/env python3
"""
Pandas project
By Ced
"""
import pandas as pd


def from_numpy(array):
    """
    from numpy to array
    name each column alphabetically
    return df
    """
    df = pd.DataFrame()
    for i in range(array.shape[1]):
        df[chr(65+i)] = array[:, i]

    return df
