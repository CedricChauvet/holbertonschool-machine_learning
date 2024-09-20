#!/usr/bin/env python3
"""
This module contains the function uni_bleu
calculates the unigram BLEU score for a sentence
"""
import numpy as np
from collections import Counter

def ngram_bleu(references, sentence, n):
    """
    calculates the n-gram BLEU score for a sentence
    param references: list of reference translations
    param sentence: list containing the model proposed sentence
    returns: the n-gram BLEU score
    """

    # Vérifier si la phrase est assez longue pour contenir le n-gramme
    if len(sentence) < n:
        return 0
    # Créer une liste de tous les n-grammes dans la phrase
    ngrams_in_sentence = [tuple(sentence[i:i+n]) for i in range(len(sentence) - n + 1)]
    # Utiliser Counter pour compter les occurrences de chaque n-gramme
    candidate_counts = Counter(ngrams_in_sentence)
    
    # count ngram in sentence
    count = sum(candidate_counts.values())


    # Compter les n-grammes dans les références
    reference_counts = Counter()
    for reference in references:
        # Créer les n-grammes pour chaque référence
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
        # Mettre à jour le compteur avec le maximum pour chaque n-gramme
        for ngram in set(ref_ngrams):
            reference_counts[ngram] = max(reference_counts[ngram], ref_ngrams.count(ngram))
        print('reference_counts', reference_counts)
    print("ngram in sentence", ngrams_in_sentence[0])
    print('ngram_counts', sum(candidate_counts.values()))
    
    # Calculer le nombre de n-grammes correspondants
    clipped_count = sum(min(count, reference_counts[ngram]) for ngram, count in candidate_counts.items())
    print('clipped_count', clipped_count)

    precision = clipped_count / count
    
    return precision