#!/usr/bin/env python3
"""
Attention project
By Ced
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    This class create a decoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        super(Decoder, self).__init__()
        self.dm = dm
        self.N = N
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        call method: return output of de decoder, same shape as x
        """
        embedding = self.embedding(x)
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        positional_encoding = self.positional_encoding[:x.shape[1], :]
        x = embedding + positional_encoding
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, encoder_output,
                      training, look_ahead_mask,
                      padding_mask)
        return x
