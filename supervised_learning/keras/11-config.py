#!/usr/bin/env python3
"""
this projet is about keras, learning with exercices,
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    task 11, save config with json
    """

    # get network into json and save in filename
    model_json = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)


def load_config(filename):
    """
    load a config
    """

    # load json and create model
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = K.models.model_from_json(loaded_model_json)
    return loaded_model
