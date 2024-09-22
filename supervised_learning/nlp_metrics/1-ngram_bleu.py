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
    ngrams_in_sentence = [tuple(sentence[i:i+n]) for i
                          in range(len(sentence) - n + 1)]
    # Utiliser Counter pour compter les occurrences de chaque n-gramme
    candidate_counts = Counter(ngrams_in_sentence)

    # count ngram in sentence
    count = sum(candidate_counts.values())

    # Compter les n-grammes dans les références
    reference_counts = Counter()
    for reference in references:
        # Créer les n-grammes pour chaque référence
        ref_ngrams = [tuple(reference[i:i+n]) for i
                      in range(len(reference) - n + 1)]
        # Mettre à jour le compteur avec le maximum pour chaque n-gramme
        for ngram in set(ref_ngrams):
            reference_counts[ngram] = max(reference_counts[ngram],
                                          ref_ngrams.count(ngram))
    #    print('reference_counts', reference_counts)
    # print("ngram in sentence", ngrams_in_sentence[0])
    # print('ngram_counts', sum(candidate_counts.values()))

    # Calculer le nombre de n-grammes correspondants
    clipped_counts = {ngram: min(count, reference_counts[ngram])
                      for ngram, count in candidate_counts.items()}
    count_clip = sum(clipped_counts.values())
    precision = count_clip / count

    r = find_closest(references, sentence)
    ref_closest = len(references[r])

    if len(sentence) < ref_closest:
        BP = np.exp(1 - (ref_closest / len(sentence)))
    else:
        BP = 1
    return BP * precision


def find_closest(references, sentence):
    """
    param references: list of reference translations
    param sentence: list containing the model proposed sentence
    return: the closest reference length to the sentence
    """
    ref_len = []
    for ref in references:
        ref_len.append(abs(len(ref) - len(sentence)))

    return ref_len.index(min(ref_len))
