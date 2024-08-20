#!/usr/bin/env python3
"""
Project auto encoders
Bu Ced+
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims is an integer containing the dimensions of the model input
    
    hidden_layers is a list containing the number of nodes for each hidden layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space representation
    """

    # création de l'encodeur
    encoder = keras.Sequential()
    encoder.add(keras.Input(shape=(input_dims,)))
    for n in hidden_layers:
        encoder.add(keras.layers.Dense(n, activation="relu"))
    encoder.add(keras.layers.Dense(latent_dims, activation="relu"))

    # création du décodeur
    decoder = keras.Sequential()
    decoder.add(keras.Input(shape=(latent_dims,)))
    for n in hidden_layers[::-1]:
        decoder.add(keras.layers.Dense(n, activation="relu"))
    decoder.add(keras.layers.Dense(input_dims, activation="sigmoid"))
    




    auto = keras.Sequential()
    auto.add(encoder)
    auto.add(decoder)
    
    
    # # Compiler le modèle
    encoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    # Afficher le résumé du modèle
   
   
    # # Compiler le modèle
    decoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    # Afficher le résumé du modèle
   
    
    # # Compiler le modèle
    auto.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    # Afficher le résumé du modèle
   

    return encoder, decoder, auto