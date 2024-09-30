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
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)
        
    def tokenize_dataset(self, data):
        """
        Instance method
        """

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        # for pt, en in self.data_train.take(1):
        #     print("Portuguese: ", pt.numpy().decode('utf-8'))
        #     print("English:   ", en.numpy().decode('utf-8'))
    
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Instance method
        """
        return None
    
    


    def display_text(slef, pt, en):
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')
        print("Portugais:", pt_text)
        print("Anglais  :", en_text)
        print()
