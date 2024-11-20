#!/usr/bin/env python3
"""
Pandas project
By Ced
"""


def fill(df):
    """
    more ways to fill missing values
    """
    del df['Weighted_Price']
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Close'] = df['Close'].fillna(method='ffill')
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])
    return df

"""
version claude
def fill(df):
    df = df.drop('Weighted_Price', axis=1)
    
    # Remplir les volumes avec 0
    df[['Volume_(Currency)', 'Volume_(BTC)']] = df[['Volume_(Currency)', 'Volume_(BTC)']].fillna(0)
    
    # Forward fill pour Close puis utiliser Close pour les autres colonnes
    df['Close'] = df['Close'].fillna(method='ffill')
    cols_to_fill = ['High', 'Low', 'Open']
    df[cols_to_fill] = df[cols_to_fill].fillna(df['Close'].values[:, None])
    
    return df

"""
