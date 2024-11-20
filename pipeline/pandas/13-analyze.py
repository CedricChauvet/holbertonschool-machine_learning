#!/usr/bin/env python3
"""
Pandas project
By Ced
"""


def analyze(df):
    """
    use describe to analyze the data
    """
    df = df[['Open', 'High', 'Low',  'Close', 'Volume_(BTC)',
             'Volume_(Currency)', 'Weighted_Price']]

    df2 = df.describe()
    return df2


# works, but longer...
# df1 = pd.DataFrame([df.count(), df.mean(), df.std(),
# df.min()], index=['count', 'mean', 'std', 'min'])
