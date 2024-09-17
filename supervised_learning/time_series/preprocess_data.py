import pandas as pd
import matplotlib.pyplot as plt

# Lecture du fichier CSV avec parse_dates
df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')


# Transformer le timestamp en date avec heure
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('Timestamp', inplace=True)
# Échantillonnage des données (une ligne sur 15)
df = df.iloc[-1440::15]

# Sélection et renommage des colonnes d'intérêt
df = df[[ 'Open', 'Volume_(BTC)', 'Weighted_Price']]
df.rename(columns={'Volume_(BTC)': 'Volume'}, inplace=True)

# Affichage des 10 premières lignes
print(df.head(50))

# Sauvegarde en CSV
df.to_csv('output.csv')

