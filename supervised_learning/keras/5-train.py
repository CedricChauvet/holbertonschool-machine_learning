#!/usr/bin/env python3
""" this projet is about keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):
    """
    this is the task 5, train a model adding a validation data
    """

    if validation_data is None:
        history = network.fit(
    x=data
    , y=labels
    , batch_size=batch_size
    , epochs=epochs
    ,verbose=False
        )    

    else:
        history = network.fit(
        x=data
        , y=labels
        , validation_data= validation_data
        , batch_size=batch_size
        , epochs=epochs
        )

        
    if verbose is False:
        valuesloss = [float("{:.5f}".format(x))
                      for x in history.history["loss"]]
        valueaccurate = [float("{:.5f}".format(x))
                         for x in history.history["accuracy"]]
    
    return history
