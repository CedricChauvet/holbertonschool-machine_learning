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

def autoencoder(input_dims, hidden_layers, latent_dims):
    # Encoder
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input
    
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.models.Model(inputs=encoder_input, outputs=latent)
    
    # Decoder
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input
    
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_output)
    
    # Autoencoder (encoder + decoder)
    auto_input = encoder_input
    auto_output = decoder(encoder(auto_input))
    auto = keras.models.Model(inputs=auto_input, outputs=auto_output)
    auto.summary()
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
