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
    build and train, word2vec model
    """
    model = Word2Vec()
    model.vector_size = 100
    model.min_count = 1
    model.window = 5
    model.negative = 5
    model.sg =  0 if cbow else 1
    model.epoch = 5
    model.seed = 0
    model.workers = 1
    model.build_vocab(common_texts)
    model.corpus_count = len(common_texts)
    model.train(common_texts, total_examples=model.corpus_count, epochs=model.epoch)

    return model