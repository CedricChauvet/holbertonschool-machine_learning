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
    _, seq_len_out = target.shape

    # tf.math.equal(inputs, 0) :
    # Cette partie compare chaque élément du tenseur inputs
    # à zéro1. Elle renvoie un tenseur booléen de même forme
    # que inputs, où chaque élément est True si la valeur
    # correspondante dans inputs est égale à zéro, et False sinon.

    # tf.cast(..., tf.float32) :
    # Cette fonction convertit le tenseur booléen résultant en un
    # tenseur de nombres flottants1. Les valeurs True deviennent
    # 1.0 et les valeurs False deviennent 0.0.
    # Créer le masque

    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, None, None, :]

    # tf.linalg.band_part
    # Copy a tensor setting everything outside a central band
    # in each innermost matrix to zero.

    mask_lower = (tf.linalg.band_part(tf.ones((batch_size, 1,
                                               seq_len_out, seq_len_out)),
                                      0, -1)
                  - tf.linalg.band_part(tf.ones((batch_size, 1,
                                                 seq_len_out, seq_len_out)),
                                        0, 0))
    encoder_target = tf.cast(tf.math.equal(target, 0), tf.float32)
    encoder_target = encoder_target[:, None, None, :]

    # mask_lower_bool = tf.cast(mask_lower, tf.bool)

    # print("shape(mask_lower)", mask_lower.shape)
    # print("shape(encoder_mask)", encoder_mask.shape)
    combined_mask = tf.maximum(encoder_target, mask_lower)
    # print("shape(combined_mask)", combined_mask.shape)
    # mask_lower = mask_lower[batch_size, None, :, :]

    return encoder_mask, combined_mask, encoder_mask
