#!/usr/bin/env python3
"""
Pandas project
By Ced
"""
import pandas as pd
import numpy as np


def from_numpy(array):
    """
    from numpy to array
    name each column alphabetically
    return df
    """
    df = pd.DataFrame()
    for i in range(array.shape[0]):
        df[chr(65+i)] = array[i, :]

    return df
