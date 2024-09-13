import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('output_btc0.csv')
df = df.dropna()  # Supprimer les valeurs manquantes
dataset = df.values
dataset = dataset.astype('float32') 

X = np.delete(dataset, 3, axis=1)  # Supprimer la 3ème colonne
y = dataset[:, 3]  


print(np.isnan(X).sum(), np.isinf(X).sum())
print(np.isnan(y).sum(), np.isinf(y).sum())


#  using MinMaxScaler to scale the data, X the whole data and y the third column
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Fenêtrage des données (séries temporelles)
seq_size = 24  # Longueur des séquences
def create_sequences(X, y, seq_size):
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


# print("len", len, "train_size", train_size)
# print("x_train", X_train.shape, "y_train", y_train.shape)
# print("x_val", X_val.shape, "y_val", y_val.shape)


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

# modele classique
seq_size= 24
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_size, 6)))
model.add(Dense(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)


pred_train = model.predict(X_train)
print("pred_train", pred_train[10])
#train_predictions = train_predictions.reshape(-1, 1)
# print("train_predictions", train_predictions.shape)
#test_predictions = test_predictions.reshape(1, -1)

pred_train_unscaled =  scaler_y.inverse_transform(pred_train) 
#test_predictions = scaler.inverse_transform(test_predictions)
y_train = scaler_y.inverse_transform(y_train)

# y_pred = scaler.inverse_transform(x_train_scaled)

print("y_train", y_train.shape)
#pred_train_unscaled = pred_train_unscaled.reshape(-1, 1)
print("y train_pred", pred_train_unscaled.shape)

plt.figure(figsize=(10, 6))
plt.plot(pred_train_unscaled) 
plt.plot(y_train, label='y')
plt.show()        






# # Tracer l'historique d'entraînement
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Perte d\'entraînement')
# plt.plot(history.history['mae'], label='MAE d\'entraînement')
# plt.title('Historique d\'entraînement du modèle')
# plt.xlabel('Époque')
# plt.ylabel('Perte / MAE')
# plt.legend()
# plt.show()

# 

# Tracer quelques prédictions
# plt.figure(figsize=(10, 6))
# plt.plot(y_batch.numpy(), label='Réel')
# plt.plot(predictions, label='Prédit')
# plt.title('Prédiction du prix du Bitcoin')
# plt.xlabel('Pas de temps')
# plt.ylabel('Prix normalisé')
# plt.legend()
# plt.show()
