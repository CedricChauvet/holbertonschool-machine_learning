#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import tensorflow as tf

class RNNEncoder:
    """RNN encoder class"""
    def __init__(self, vocab, embedding, units, batch):
        """Constructor"""
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.units = units
        self.gru = tf.keras.layers.GRU(units, recurrent_initializer='glorot_uniform')e
        self.batch = batch

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell to a tensor of zeros"""
        return np.zeros((self.batch, self.units))

    def __call__(self, x, initial):
        """Forward prop"""
        embeddings = self.embedding[x]
        outputs, hidden = self.gru.forward(initial, embeddings)
        return outputs, hidden