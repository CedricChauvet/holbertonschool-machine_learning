#!/usr/bin/env python3
"""
Project auto encoders
Bu Ced+
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    cet autoencodeur utilise des filtres convolutif
    suive de maxpool pour le retrecissement et de upsampling
    La difficulté est dans le padding pour respecter les dimensions
    on se retrouve avec des dimensions impaires
    """

    # creation de l'encodeur, 1ere moitié
    encoder = keras.Sequential()
    encoder.add(keras.Input(shape=(input_dims)))
    for n in filters:
        encoder.add(keras.layers.Conv2D(n, activation='relu', kernel_size=(3, 3), padding='same'))
        encoder.add(keras.layers.MaxPooling2D((2, 2), padding='same'))

    # création du decoder et de la recontruction de l'image
    decoder = keras.Sequential()
    for n in filters[::-1][0:2]:
        decoder.add(keras.layers.Conv2D(n, activation='relu', padding='same', kernel_size=(3, 3)))
        decoder.add(keras.layers.UpSampling2D((2, 2)))

    decoder.add(keras.layers.Conv2D(filters[-1
                                            ], activation='sigmoid', padding='valid', kernel_size=(3, 3)))
    decoder.add(keras.layers.UpSampling2D((2, 2)))
    decoder.add(keras.layers.Conv2D(input_dims[2], activation='sigmoid', padding='same', kernel_size=(3, 3)))

    # utiliser add permet de stacker les layers, pratique
    # mais attention au dimensions entre 2 sequentials
    auto = keras.Sequential()
    auto.add(encoder)
    auto.add(decoder)

    # Compile the autoencoder, binary crossentrepy pour les 10 labels
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    # auto.summary()
    return encoder, decoder, auto
