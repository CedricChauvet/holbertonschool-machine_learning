#!/usr/bin/env python3
"""
Attention project
By Ced
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.\
            LayerNormalization(axis=-1, epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.\
            LayerNormalization(axis=-1, epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.\
            LayerNormalization(axis=-1, epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        call function,
        returns: tensor of shape (batch, target_seq_len, dm)   
        """


        out1, _ = self.mha1(x, x, x, padding_mask)
        drop1 = self.dropout1(out1, training=training)
        norm1 = self.layernorm1(x + drop1)
    
        # decoder output into MHA_2
        out3, _ = self.mha2(norm1, encoder_output, encoder_output, look_ahead_mask)
        out3 = self.dropout2(out3, training=training)
        norm2 = self.layernorm2(out3 + norm1)

        # feed forward
        feed_input = self.dense_hidden(norm2)
        feed_output = self.dense_output(feed_input)
        feed_drop = self.dropout3(feed_output, training=training)

        decoder_output = self.layernorm3(norm2 + feed_drop)
      
        return decoder_output