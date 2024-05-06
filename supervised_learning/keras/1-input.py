#!/usr/bin/env python3
""" first program of Keras, to be continued"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ that builds a neural network with the Keras library
    be careful about  lmbtha and keep prob, that ensur the dropout
    """

    # define the keras model
    
# Entrée du modèle
    inputs = K.Input(shape=(nx,))
    i = 0
# Création des couches cachées
    x = inputs
    for lay, act in zip(layers, activations):
        i += 1
        x = K.layers.Dense(lay, activation=act,
                  kernel_regularizer=K.regularizers.l2(lambtha))(x)
        
        if i < len(layers):
            x = K.layers.Dropout(1 - keep_prob)(x)
         
 
    model = K.Model(inputs=inputs, outputs=x)
    
    return model
