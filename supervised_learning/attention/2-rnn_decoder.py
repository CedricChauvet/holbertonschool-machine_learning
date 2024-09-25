#!/usr/bin/env python3
"""
Attention project
by Ced
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

class RNNDecoder(tf.keras.layers.Layer):
    """
    build a decoder class for RNN
    """

    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
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
        c = SelfAttention(s_prev.shape[1])(s_prev, hidden_states)[0]
        print("c shape", c.shape)
        print("X shape", x.shape)
        concat1 = tf.concat((c, x), axis=1)


        #print("concat", concat.shape)
        y,s = self.gru(concat1, initial_state=s_prev)
        return y, s
