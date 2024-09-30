#!/usr/bin/env python3
"""
Transformer Applications project
By Ced
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers

class Dataset():
    def __init__(self):
        """
        Class constructor
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        self.tokenizer_pt = transformers.AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.tokenizer_en = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_dataset(self, data):
        """
        Instance method
        """
        # tokenizer_pt = transformers.AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        # tokenizer_en = transformers.AutoTokenizer.from_pretrained('bert-base-uncased

        return self.tokenizer_pt, self.tokenizer_en


    def encode(self, pt, en):
        """
        Instance method
        """
        return None
    
    

