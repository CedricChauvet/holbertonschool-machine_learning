#!/usr/bin/env python3
"""
Attentin project
By Ced
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Build a decoder class for RNN with attention mechanism
    """

    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Calling method
        Returns:
        - y: tensor of shape (batch, vocab) containing the output word
             as a one-hot vector in the target vocabulary
        - s: tensor of shape (batch, units) containing the
              new decoder hidden state
        """
        # Apply embedding to input
        x = self.embedding(x)

        # Apply attention
        context_vector, _ = self.attention(s_prev, hidden_states)

        # Concatenate attention context vector and embedded input
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Pass through GRU
        x, s = self.gru(x)

        # Reshape output and pass through final dense layer
        x = tf.reshape(x, (-1, x.shape[2]))
        y = self.F(x)

        return y, s
