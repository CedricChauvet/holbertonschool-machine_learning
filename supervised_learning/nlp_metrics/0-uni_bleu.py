#!/usr/bin/env python3
"""
This module contains the function uni_bleu
calculates the unigram BLEU score for a sentence
"""
import numpy as np

def uni_bleu(references, sentence):
    """
    param references: list of reference translations
    param sentence: list containing the model proposed sentence
    return: the unigram BLEU score
    """

    count = 0
    for word in sentence:
        refs=[]
        # print('word', word, sentence.count(word)) 
        for ref in references:
        #     print('ref', ref)   
              refs.append(ref.count(word))    
        # print('refs', refs,"max", max(refs), "\n")

        count += min(sentence.count(word), max(refs))

    P1 = count/len(sentence)  
    print('P1', P1) 
    
    r = find_closest(references, sentence)
    ref_closest = len(references[r])

    if len(sentence) < ref_closest:
        BP = np.exp(1 - ( ref_closest / len(sentence)))
    else: 
        BP = 1

    # print('len(sentence)', len(sentence), 'len(references)', len(references[0]))

    #print('len(sentence)', len(sentence), 'len(references)', len(references[:]))
    # print('BP', BP)
    return P1 * BP

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