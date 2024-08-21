#!/usr/bin/env python3
"""
Project auto encoders
Bu Ced+
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):

    # Encoder
    # creation d'une input pour les images
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input  # variable modulable

    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu', kernel_regularizer=keras.regularizers.L1(lambtha))(x)

    # premier modele
    encoder = keras.models.Model(inputs=encoder_input, outputs=latent)

    # Decoder
    # 2 eme input positionné derriere le botleneck
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    # sortie du decodeur
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_output)

    # Autoencoder (encoder + decoder)
    auto_input = encoder_input   # mise en cache
    auto_output = decoder(encoder(auto_input))

    # dernier modele
    auto = keras.models.Model(inputs=auto_input, outputs=auto_output)
    # auto.summary()
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
