#!/usr/bin/env python3
"""
Attention project
By Ced
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder (tf.keras.layers.Layer):
    """
    This class create an encoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()
        self.N = N  # number of blocks
        self.dm = dm  # dimensionality of the model
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Call method, return a tensor of
        shape (batch, input_seq_len, dm)
        containing the encoder output
        """
        seq_len = tf.shape(x)[1]

        # Word Embedding
        embedding = self.embedding(x)  # (batch, input_seq_len, dm)

        # Scale embedding
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding
        pos_encoding = self.positional_encoding[:seq_len, :]
        encoder_input = embedding + pos_encoding

        # Apply dropout
        x = self.dropout(encoder_input, training=training)

        # Pass through each encoder block
        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        return x
