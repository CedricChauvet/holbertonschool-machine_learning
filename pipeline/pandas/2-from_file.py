#!/usr/bin/env python3
"""
Pandas project
By Ced
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    read a file (csv) to a dataframe
    """
    file = pd.read_csv(filename, delimiter=delimiter)
    return file
