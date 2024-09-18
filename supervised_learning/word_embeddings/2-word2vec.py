#!/usr/bin/env python3
"""
NLP project
by Ced
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    build and train, word2vec model
    """
    model = gensim.models.Word2Vec()
    model.vector_size = vector_size
    model.min_count = min_count
    model.window = window
    model.negative = negative
    model.sg =  0 if cbow else 1
    model.epochs = epochs
    model.seed = seed
    model.workers = workers
    model.build_vocab(sentences)
    model.corpus_count = len(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

    return model