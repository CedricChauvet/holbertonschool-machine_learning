#!/usr/bin/env python3
"""
code to make time series data in a dataset, variable are resized
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')

df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s').dt.strftime('%Y/%m/%d - %H:%M')
# give date as index
df.set_index("Timestamp", inplace=True)
print(df.dtypes)

# prendre la fin du document avec un intervalle de 5 minutes
df = df[-100000::60]
print ("taille de la donnée", len(df))
df.to_csv('output_btc0.csv', index=False)


# print("x shape", X.shape)
# print("y shape", y.shape) 
# Diviser en ensembles d'entraînement et de test
# train_size = int(len(X) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# np.savez("./out_btc.npz", X, y)




# # Créer des tf.data.Dataset
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))



# # Créer des tf.data.Dataset
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# # Préparer les datasets pour l'entraînement et le test
# # batch_size = 32
# # train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# # test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# # Sauvegarder les datasets
# tf.data.Dataset.save(train_dataset, "train_dataset")
# tf.data.Dataset.save(test_dataset, "test_dataset")