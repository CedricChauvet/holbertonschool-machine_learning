#!/usr/bin/env python3
"""
whole transformer model
"""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    to be filled
    """
    data = Dataset(batch_size, max_len)
    vocab_size =  data.tokenizer_pt.vocab_size + 2
    transformer = Transformer(N, dm, h, hidden, vocab_size, vocab_size,
                              max_len, max_len)
    print("transformer", type(transformer))