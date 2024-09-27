#!/usr/bin/env python3
"""
Attention project
By Ced
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    perform multi head attention:
    """
    
    def __init__(self, dm, h):
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)
    
    def __call__(self, Q, K, V, mask= None):
        """
        Call method
        """
        # Q shape (batch seq_len_q, dk)
        print("q", Q.shape)
        sdp_attention(self.Wq * Q, self.Wk * K,self.Wv * V, mask)