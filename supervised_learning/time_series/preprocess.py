#!/usr/bin/env python3
"""
code to make time series data in a dataset, variable are resized
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')

df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')\
                                 .dt.strftime('%Y/%m/%d - %H:%M')
# give date as index
df.set_index("Timestamp", inplace=True)
print(df.dtypes)

# take the end of the document with a 60-minute interval
df = df[-100000::60]
print("taille de la donnée", len(df))
df.to_csv('article_output_1.csv', index=False)
