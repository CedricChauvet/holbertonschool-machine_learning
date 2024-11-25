#!/usr/bin/env python3
"""
Pandas project
By Ced
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator

import pandas as pd
import datetime
from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df = df.rename(columns={'Timestamp': 'Date'})

df['Date'] = pd.to_datetime(df['Date'], unit='s')
df['Date'] = df['Date'].dt.date
df = df.loc[df['Date'] >= pd.to_datetime('2017-01-01').date()]

df['Close'] = df['Close'].fillna(method='ffill')
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)

df2 = pd.DataFrame()


df2['High'] = df.groupby('Date')['High'].max()
df2['Low'] = df.groupby('Date')['Low'].min()
df2['Open'] = df.groupby('Date')['Open'].mean()
df2['Close'] = df.groupby('Date')['Close'].mean()
df2['Volume_(BTC)'] = df.groupby('Date')['Volume_(BTC)'].sum()
df2['Volume_(Currency)'] = df.groupby('Date')['Volume_(Currency)'].sum()

# affichage des données pour vérification
print(df2)

"""
Partie graphqie
Plot the data de janvier 2017 à janvier 2019
"""


# Formateur personnalisé pour afficher l'année uniquement en janvier
def custom_formatter(x, pos=None):
    """
    pour le plot.
    permet d'afficher l'année uniquement en janvier
    """
    date = matplotlib.dates.num2date(x)
    if date.month == 1:
        return date.strftime('%b \n %Y')
    return date.strftime('%b')


df2.plot()

ax = plt.gca()
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

ax.xaxis.set_minor_locator(MonthLocator([2, 3, 5, 6, 8, 9, 11, 12]))

ax.tick_params(which='minor', length=2)
ax.set_xlim(datetime.date(2017, 1, 1), datetime.date(2019, 1, 1))
plt.xlabel('Date')
plt.show()
