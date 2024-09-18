#!/usr/bin/env python3
"""
NLP project
by Ced
"""
import gensim
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential

def gensim_to_keras(model):

    keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array    
    index_to_key = keyed_vectors.index_to_key  # which row in `weights` corresponds to which word?

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True,
    )

    return layer
    
