#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import tensorflow as tf




def l2_reg_create_layer(prev, n, activ, lambtha):
    


    L2_layer = tf.keras.layers.Dense(units=n,activation=activ,
           kernel_regularizer=tf.keras.regularizers.L2(
    l2=lambtha))(prev)
    
    return L2_layer    


#kernel_regularizer='l2', activity_regularizer='l2'