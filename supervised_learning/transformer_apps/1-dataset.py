#!/usr/bin/env python3
"""
Transformer Applications project
By Ced
"""
import tensorflow_datasets as tfds
import transformers


class Dataset():
    """
    this class loads and preps a dataset for machine translation
    """
    def __init__(self):
        """
        Class constructor
        """
        self.data_train, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                                       split='train', with_info=True,
                                       as_supervised=True)

        self.data_valid, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                                       split='validation', with_info=True,
                                       as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en =\
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Instance method
        """

        tokenizer_pt = transformers.AutoTokenizer.\
            from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.\
            from_pretrained('bert-base-uncased')

        def get_training_corpus_en():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        def get_training_corpus_pt():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')
        #  How to tokenize a sentence
        #  sentences = "i love you madly"
        #  tokens = tokenizer_en.tokenize(sentences)
        #  print(tokens)

        tokenizer_pt = tokenizer_pt.\
            train_new_from_iterator(get_training_corpus_pt(), vocab_size=8192)
        tokenizer_en = tokenizer_en.\
            train_new_from_iterator(get_training_corpus_en(), vocab_size=8192)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Instance method
        """

        nouveau_cls_id = 8192
        nouveau_sep_id = 8193

        # Exemple de tokenization manuelle avec vos propres IDs
        pt_tokens = [nouveau_cls_id] + self.tokenizer_pt.encode(pt.numpy().decode('utf-8'), add_special_tokens=False) + [nouveau_sep_id]
        en_tokens = [nouveau_cls_id] + self.tokenizer_en.encode(en.numpy().decode('utf-8'), add_special_tokens=False) + [nouveau_sep_id]
        #print("encode en", self.tokenizer_en.encode(en))
        return pt_tokens, en_tokens
