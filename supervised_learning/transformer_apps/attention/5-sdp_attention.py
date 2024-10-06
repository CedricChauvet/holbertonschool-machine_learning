#!/usr/bin/env python3
"""
Attention project
By Ced
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Scaled Dot Product Attention is a key component of
    the transformer architecture
    param Q: tensor with shape (..., seq_len_q, dk) containing the query matrix
    param K: tensor with shape (..., seq_len_v, dk) containing the key matrix
    param V: tensor with shape (..., seq_len_v, dv) containing the value matrix
    param mask is always None
    Returns: output, weights
        outputa tensor with its last two dimensions as (..., seq_len_q, dv)
            containing scaled dot product attention
        weights a tensor with its last two dimensions as
            (..., seq_len_q, seq_len_v) containing the attention weights
    """

    # Calcul du produit scalaire entre Q et K transposé
    matmul_QK = tf.matmul(Q, K, transpose_b=True)

    # Mise à l'échelle par la racine carrée de la dimension des clés
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_scores = matmul_QK / tf.math.sqrt(dk)

    # Appliquer un masque (optionnel)
    if mask is not None:
        scaled_scores += (mask * -1e9)  # -1e9: -1*10^9 for softmax

    # Appliquer Softmax aux scores pour obtenir les poids d'attention
    attention_weights = tf.nn.softmax(scaled_scores, axis=-1)

    # Multiplication des poids d'attention avec les valeurs V
    output = tf.matmul(attention_weights, V)

    return output, attention_weights
