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
        c , _= SelfAttention(s_prev.shape[1])(s_prev, hidden_states)
        x = self.embedding(x)
   
   
        # Concatenate attention context vector and embedded input
        x = tf.concat([tf.expand_dims(c, 1), x], axis=-1)
        
        # Pass through GRU
        output, state = self.gru(x, initial_state=s_prev)
        
        # Reshape output
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # Pass through final dense layer
        y = self.F(output)
        
        return y, state