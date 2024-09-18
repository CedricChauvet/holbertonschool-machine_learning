#!/usr/bin/env python3
"""
NLP project,
google search engine
by Ced
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    converts a gensim word2vec model to a trainable keras layer
    """

    keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array

    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True,
    )

    return layer
