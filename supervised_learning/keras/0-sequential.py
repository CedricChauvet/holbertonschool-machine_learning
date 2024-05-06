#!/usr/bin/env python3
import tensorflow.keras as K
""" first program of Keras, to be continued"""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ that builds a neural network with the Keras library"""
    # define the keras model    
    num_layers = 3
    num_neurons = 64
    activation = 'relu'

# Définition du modèle
    model = K.Sequential()
    for lay,act in zip(layers,activations):
        model.add(K.layers.Dense(lay, activation=act,input_shape=(nx,), kernel_regularizer=K.regularizers.l2(lambtha),))
        model.add(K.layers.Dropout(1 - keep_prob))
        input_shape = (nx,)  # La taille de l'entrée change après la première couche

    return model

    """
    model = K.Sequential()
    model.add(K.Input(shape=(nx,)))
    model.add(layers.Dense(2, activation="relu", name="layer1"))
    model.add(layers.Dense(3, activation="relu", name="layer2"))
    model.add(layers.Dense(4, name="layer3"))
    return model
      
    for lay,act in zip(layers,activations):
        model.Dense(lay, activation=act)
    return model
    """    