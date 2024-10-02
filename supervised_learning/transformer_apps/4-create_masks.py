#!/usr/bin/env python3
"""
Transformer Applications project
By Ced, performing NLP tasks, enhanced
"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    method for masking training instances
    and validating instances
    """
    batch_size, seq_len_in = inputs.shape
    _, seq_len_out = target.shape # target.shape
    # tf.linalg.band_part
    # Copy a tensor setting everything outside a central band
    # in each innermost matrix to zero.
    
    # encoder mask
    tf_ones = tf.zeros((batch_size, 1, 1,seq_len_in))
    encoder_mask = tf.linalg.band_part(tf_ones, 0, -1)

    combined_mask = None
    decoder_mask = None 

    
    return encoder_mask, combined_mask, decoder_mask
