#!/usr/bin/env python3
"""
code to make time series data in a dataset, variable are resized
"""
import pandas as pd
import numpy as np
df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')

print("Number of corrupted Nan values", np.isnan(df).sum())
df = df.dropna()  # Supprimer les valeurs manquantes

df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')
# Deuxième ligne : Formater le datetime en chaîne de caractères
df["Timestamp"] = df["Timestamp"].dt.strftime('%Y/%m/%d - %H:%M')

# 1. give date as index
df.set_index("Timestamp", inplace=True)

# 2. Appliquez la conversion float32 uniquement à ces colonnes
list_columns = ["Open", "High", "Low", "Close", "Volume_(BTC)",
                "Volume_(Currency)", "Weighted_Price"]
df[list_columns] = df[list_columns].astype('float32')
print(df.dtypes)  # print the type of data inside df

# 3. prendre la fin du document avec un intervalle de 60 minutes
df = df[-100000::60]

print("taille de la donnée", len(df))
print("interval des données du", df.index[0], "au", df.index[-1])

df.to_csv('output_btc0.csv', index=True)
