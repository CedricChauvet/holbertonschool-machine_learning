#!/usr/bin/env python3
"""
Pandas project
By Ced
"""
import pandas as pd


def rename(df):
    """
    doing some renaming, and converting the timestamp to datetime
    selecting only the columns Datetime and Close
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
