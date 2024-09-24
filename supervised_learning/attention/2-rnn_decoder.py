#!/usr/bin/env python3
"""
Attention project
by Ced
"""
import tensorflow as tf


class RNNDecoder(tf.keras.layers.Layer):
    """
    build a decoder class for RNN
    """

    def __init__(self, vocab, embedding, units, batch):
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                recurrent_initializer='glorot_uniform',
                                return_sequences=True,
                                return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        calling method 
        returns y the tensor of shape (batch, vocab) containing
        the output word as a one hot vector in the target vocabulary
        and s is a tensor of shape (batch, units) containing the new
        decoder hidden state
        """

        return y, s
