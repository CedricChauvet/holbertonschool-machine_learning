#!/usr/bin/env python3
"""
Attention project
by Ced
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    self attention class
    inherits from a keras layer
    """

    def __init__(self, units):
        """Constructor"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)  # sure?

    def call(self, s_prev, hidden_states):
        """
        call for return context and weights
        """
        s_prev = tf.expand_dims(s_prev, 1)
        decoder = self.W(s_prev)
        # print("decoder",decoder.shape)
        encoder = self.U(hidden_states)
        # print("encoder",encoder.shape)
        score = self.V(tf.nn.tanh(encoder + decoder))
        # Calcul des poids d'attention
        weights = tf.nn.softmax(score, axis=1)

        # Calcul du vecteur de contexte
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
