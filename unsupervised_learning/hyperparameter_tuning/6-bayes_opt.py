from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les images
x_train = x_train / 255
x_test = x_test / 255

# Reshaper les images pour qu'elles aient la forme (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convertir les labels en matrices binaires
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def create_model(params):
    dropout_rate = params[0][0]
    l2_weight = params[0][1]
    learning_rate = 0.001 # params[0][2]
    batch_size = 32 # params[0][3]
    layer_one = 8 # params[0][4]
    layer_two = 8 # params[0][5]

    # Construire le modèle
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(layer_one, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(layer_two, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(10, activation="softmax"),
        ])
    model.compile(
        regularizer = tf.keras.regularizers.L2(l2= l2_weight), 
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def objective_function(params):
    model = create_model(params)
    history = model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))
    validation_accuracy = history.history['val_accuracy'][-1]
    return -validation_accuracy

# Définir les bornes de l'espace de recherche
bounds = [
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.2, 0.7)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (0.000001, 1000)},
    
]

# Créer l'objet d'optimisation bayésienne
optimizer = BayesianOptimization(f=objective_function, domain=bounds)

# Lancer l'optimisation
optimizer.run_optimization(max_iter=2)

# Afficher les résultats
print("Paramètres optimaux:", optimizer.X[np.argmin(optimizer.Y)])
print("Meilleure précision de validation:", -np.min(optimizer.Y))
