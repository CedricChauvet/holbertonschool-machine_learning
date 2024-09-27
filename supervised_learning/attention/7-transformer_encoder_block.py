#!/usr/bin/env python3
"""
Attention project
By Ced
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Encoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        x is tensor of shape (batch, input_seq_len, dm)
        training is a boolean to determine if the model is training
        mask is...
        returns: tensor of shape (batch, input_seq_len, dm)
        containning the block’s output
        """

        out1, Attention_W = self.mha(x, x, x, mask) 
        drop1 = self.dropout1(out1, training=training)
        out2 = self.layernorm1(x + drop1)
        out3 = self.dense_hidden(out2)
        out4 = self.dense_output(out3)

        drop2 = self.dropout2(out4, training=training)
        decoder_out = self.layernorm2(out2 + drop2)
        return decoder_out