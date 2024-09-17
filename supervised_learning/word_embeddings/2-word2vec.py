#!/usr/bin/env python3
"""
NLP project
by Ced
"""
import re
import numpy as np
from gensim.test.utils import common_texts
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    train and text word2vec model
    """

    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=True, seed=seed, negative=negative, epochs=epochs)
    return model