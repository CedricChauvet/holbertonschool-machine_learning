#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import tensorflow as tf




def l2_reg_create_layer(prev, n, activ, lambtha):
    
    
    tf.keras.regularizers.L2(
    l2=lambtha
)
    L2_layer = tf.keras.layers.Dense(n,activation=activ,
            kernel_regularizer='l2')(prev)
    return L2_layer    
