#!/usr/bin/env python3
"""
NLP project,
google search engine
by Ced
"""
import gensim


def fasttext_model(sent, vector_size=100, min_count=5,
                   negative=5, window=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    build and train, fasttext model, using gensim
    params: sent: list of sentences to be trained
    etc...
    return: model, which keep the semantic of a word
    """
    cbow = not cbow
    model = gensim.models.FastText(
        sentences=sent,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=cbow,
        negative=negative,
        seed=seed
    )

    # common_texts = gensim.test.utils.common_texts
    # Build the vocabulary
    model.build_vocab(sent)

    # Train the model for many epochs
    model.train(sent, total_examples=model.corpus_count, epochs=epochs)
    return model
