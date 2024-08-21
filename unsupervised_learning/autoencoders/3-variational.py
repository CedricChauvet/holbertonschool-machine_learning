#!/usr/bin/env python3
"""
Project auto encoders
Bu Ced+
"""
import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    ok see the vidz!!!
    """

    # Encoder
    # creation d'une input pour les images
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input  # variable modulable

    # print("conv shape", keras.backend.int_shape(x))
    
    #x = keras.layers.Flatten()(x)
    

    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    
    z_mu = keras.layers.Dense(latent_dims, name="mu")(x)
    z_sigma = keras.layers.Dense(latent_dims, name="sigma")(x)


    
    z = keras.layers.Lambda(sample_z, output_shape=(latent_dims,), name='z')([z_mu, z_sigma])

    # premier modele
    encoder = keras.models.Model(inputs=encoder_input, outputs=[z_mu, z_sigma, z], name='encoder')
    #encoder.summary()


    
    # Decoder
    # 2 eme input positionné derriere le botleneck
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    # sortie du decodeur
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_output)


    auto_input = keras.layers.Input(shape=input_dims)
    _, _, z = encoder(auto_input)
    decoded = decoder(z)
    auto = keras.models.Model(auto_input, decoded, name='autoencoder')
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

def sample_z(args):
    z_mu,z_sigma = args
    eps = keras.backend.random_normal(shape = (keras.backend.shape(z_mu)[0], keras.backend.int_shape(z_mu)[1]))
    return z_mu + keras.backend.exp(z_sigma / 2) * eps
