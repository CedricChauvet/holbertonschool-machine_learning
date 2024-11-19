#!/usr/bin/env python3
"""
Pandas project
By Ced
"""


def prune(df):
    """
    using the dropna method to remove missing values
    seen in time series forecasting project
    """

    df = df.dropna()  # Supprimer les valeurs manquantes
    return df
