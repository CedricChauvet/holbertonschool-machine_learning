#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN encoder class
       inherits from a keras layer
    """
    def __init__(self, vocab, embedding, units, batch):
        """Constructor"""
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.units = units
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.batch = batch

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to
        a tensor of zeros"""
        return tf.zeros((self.batch, self.units))

    def __call__(self, x, initial):
        """Forward prop"""
        embeddings = self.embedding(x)
        outputs, hidden = self.gru(embeddings, initial)
        return outputs, hidden
