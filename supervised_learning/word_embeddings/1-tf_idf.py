#!/usr/bin/env python3
"""
NLP project
by Ced
"""
import re
import numpy as np
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Convert a collection of raw documents to a matrix of TF-IDF features
    """

    vectorizer = TfidfVectorizer(vocabulary=vocab)
    Y = vectorizer.fit_transform(sentences).todense()

    feat = vectorizer.get_feature_names_out()
    return Y, feat
