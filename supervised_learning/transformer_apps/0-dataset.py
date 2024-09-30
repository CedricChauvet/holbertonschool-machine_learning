#!/usr/bin/env python3
"""
Transformer Applications project
By Ced
"""
import tensorflow_datasets as tfds
import transformers

class Dataset():
    def __init__(self):
        """
        Class constructor
        """
        self.data_train, _ = tfds.load('ted_hrlr_translate/pt_to_en', split='train', with_info=True, as_supervised=True)
        self.data_valid, _ = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', with_info=True, as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)
        
    def tokenize_dataset(self, data):
        """
        Instance method
        """

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Instance method
        """
        return None
    