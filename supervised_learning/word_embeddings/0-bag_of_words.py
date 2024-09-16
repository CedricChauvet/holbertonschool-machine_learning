#!/usr/bin/env python3
"""
NLP project
by Ced
"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix:
    """

    if vocab is None:
        features = set()

    for sentence in sentences:
        words = sentence.split(" ")
        for word in words:
            word = word.replace("!", "").replace(".", "")\
                   .replace(",", "").replace("?", "")\
                   .replace("'", "").replace('"', "").lower()
            features.add(word)  
    
    features = sorted(features)            
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(sentences):
        sentence = sentence.lower()
        for j, word in enumerate(features):
            c = compter(word, sentence)
            embeddings[i, j] = c    

        
        
    return embeddings, np.array(features)

def compter(word, phrase):
    """
    Compte les occurrences d'un mot spécifique dans une phrase.
    
    Args:
    word (str): Le mot à rechercher.
    phrase (str): La phrase dans laquelle chercher.
    
    Returns:
    int: Le nombre d'occurrences du mot dans la phrase.
    """
    # Crée un pattern qui recherche le mot entier
    pattern = r'\b' + re.escape(word) + r'\b'
    
    # Trouve toutes les occurrences et retourne leur nombre
    occurrences = len(re.findall(pattern, phrase, re.IGNORECASE))
    return occurrences

