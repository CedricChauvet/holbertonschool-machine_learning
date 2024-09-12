import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# btc = df[ ['Open','Close', 'Weighted_Price', 'Volume_(BTC)'] ]
# btc.plot()

df = pd.read_csv('output_btc0.csv')
# print(df)
dataset = df.values
dataset = dataset.astype('float32') 
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Convertir en tf.data.Dataset
def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), :])
        y.append(dataset[i + time_steps, 1])
    return np.array(X), np.array(y) # attetion à la sortie y[1] qui est un scalaire


  
# Définir le nombre de pas de temps (time steps) pour la séquence
time_steps = 24  # par exemple, utiliser les 60 dernières minutes pour prédire la prochaine

# Créer les ensembles X et y
n = 0.66
X, y = create_dataset(dataset, time_steps)

x_train, y_train = X[:int(n*len(X))], y[:int(n*len(X))]
x_test, y_test = X[int(n*len(X)):], y[int(n*len(X)):]
# print("eln", int(n*len(X)))
# print("x", X.shape)
# print("x_test shape", x_test.shape)
# print("x_train shape", x_train.shape)
# x_train = x_train.reshape(x_train.shape[0],1,  x_train.shape[1] )
# x_test = x_test.reshape(None,x_test.shape[1], x_test.shape[2] )

seq_size= 24
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_size, 7)))
model.add(Dense(32))
model.add(Dense(1))

# Compiler le modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraîner le modèle
history = model.fit(
    x_train, y_train, validation_data=(x_test, y_test),
    epochs=5)

train_predictions = model.predict(x_train)
test_predictions = model.predict(x_test)

train_predictions = train_predictions.reshape(-1)
test_predictions = test_predictions.reshape(-1)
print("train_predictions", train_predictions.shape)
print("y_train", y_train.shape)
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform(y_test)
y_train = scaler.inverse_transform(y_train)

"""
exit()

trainpredictplot = np.empty_like(dataset)
trainpredictplot[:,:] = np.nan
trainpredictplot[seq_size:len(train_predictions)+seq_size, :] = train_predictions

testpredictplot = np.empty_like(dataset)
testpredictplot[:,:] = np.nan
testpredictplot[len(train_predictions)+(seq_size*2)+1:len(dataset)-1, :] = test_predictions


plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainpredictplot) 
plt.plot(testpredictplot)
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
"""