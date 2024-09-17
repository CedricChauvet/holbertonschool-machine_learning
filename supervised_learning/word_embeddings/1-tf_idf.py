#!/usr/bin/env python3
"""
NLP project
by Ced
"""
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    
    """    

    if vocab is None:  # si le vocabulaire n'est pas donné
        features = set()
        for sentence in sentences:
            words = sentence.split(" ")
            for word in words:
                word = word.lower()
                word = formatter(word)
                features.add(word)

        features = sorted(features)
    else:
        features = vocab
    for i, sentence in enumerate(sentences):
        sentence = sentence.lower()
        for j, word in enumerate(features):
            tf[i,j]


    return embeddings, np.array(features)