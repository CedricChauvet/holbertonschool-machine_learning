import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error

df = pd.read_csv('article_output_1.csv')
df = df.dropna()  # Supprimer les valeurs manquantes
dataset = df.values

X = dataset  # Suppress 3rd column
y = dataset[:, 3]

# check if the data are not corrupted
# print(np.isnan(X).sum(), np.isinf(X).sum())
# print(np.isnan(y).sum(), np.isinf(y).sum())


#  using MinMaxScaler to scale the data,
#  X the whole data and y the third column
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Fenêtrage des données (séries temporelles)
seq_size = 24  # Longueur des séquences


def create_sequences(X, y, seq_size):
    """
    create sequences of seq_size, Xs and ys
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_size):
        Xs.append(X[i:i+seq_size])
        ys.append(y[i+seq_size])
    return np.array(Xs), np.array(ys)


X_seq, y_seq = create_sequences(X, y, seq_size)

# Si vous voulez une séparation 80/20
n = 0.8
len = len(X_seq)

# Division des données en train et validation (80/20)
train_size = int(n * len)
X_train, X_val = X_seq[:train_size], X_seq[train_size:]
y_train, y_val = y_seq[:train_size], y_seq[train_size:]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
                .batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
                .batch(32)

# modele classique
seq_size = 24
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_size, 7)))
model.add(Dense(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)

trainScore = math.sqrt(mean_squared_error(y_train, model.predict(X_train)))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_val, model.predict(X_val)))
print('Test Score: %.2f RMSE' % (testScore))

pred_train = model.predict(X_train)

pred_train_unscaled = scaler_y.inverse_transform(pred_train)
y_train = scaler_y.inverse_transform(y_train)

print("y_train", y_train.shape)
print("y train_pred", pred_train_unscaled.shape)

plt.figure(figsize=(10, 6))
plt.plot(pred_train_unscaled)
plt.plot(y_train)
plt.title("Prédiction de la valeur sur tout le dataset")
plt.xlabel("Heures")
plt.ylabel("Valeur de Y en dollars")
plt.legend()
plt.grid(True)
plt.show()

last_24_hours_X = X_seq[-1].reshape(1, seq_size, X.shape[1])
real_y_next_hour = y_seq[-1]  # Valeur réelle à prédire pour h+1

# Utiliser le modèle pour prédire h+1
predicted_y_next_hour = model.predict(last_24_hours_X)

# Inverser la transformation pour obtenir les valeurs originales (non scalées)
predicted_y_next_hour = scaler_y.inverse_transform(predicted_y_next_hour)
real_y_next_hour = scaler_y.inverse_transform(real_y_next_hour.reshape(-1, 1))

# Extraire les valeurs de la dernière séquence de 24 heures (non scalées)
last_24_hours_y = scaler_y.inverse_transform(y_seq[-seq_size:].reshape(-1, 1))

# Affichage des 24 dernières heures et prédiction pour h+1
plt.figure(figsize=(10, 6))

# Affichage des 24 dernières heures (valeurs réelles)
plt.plot(range(24), last_24_hours_y,
         label='24h de données (réelles)',
         color='blue')

# Affichage de la prédiction pour h+1
plt.scatter(24, predicted_y_next_hour, color='red',
            label=f'Prédiction pour h+1:\
            {predicted_y_next_hour[0][0]:.2f}')

# Affichage de la vraie valeur pour h+1
plt.scatter(24, real_y_next_hour, color='green',
            label=f'Valeur réelle pour h+1:\
            {real_y_next_hour[0][0]:.2f}')

# Labels et légende
plt.title("Prédiction de la valeur Y à h+1")
plt.xlabel("Heures")
plt.ylabel("Valeur de Y")
plt.legend()
plt.grid(True)
plt.show()
