#!/usr/bin/env python3
"""
Attention project
By Ced
"""
import tensorflow as tf
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from keras.backend import softmax
 
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    perform multi head attention:
    """
    
    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)
    

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.h, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.dm))
        return x
 
    def call(self, Q, K, V, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.Wq(Q), self.h, True)
        k_reshaped = self.reshape_tensor(self.Wk(K), self.h, True)
        v_reshaped = self.reshape_tensor(self.Wv(V), self.h, True)
        o_reshaped, attention_weights = sdp_attention(q_reshaped, k_reshaped, v_reshaped, mask)
 
 
        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.h, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)
 
        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.linear(output), attention_weights