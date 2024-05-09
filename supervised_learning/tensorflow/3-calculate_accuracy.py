#!/usr/bin/env python3
"""
2 task on tensoflow project
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_accuracy(Y, y_pred):
    # Créer un tenseur de booléens pour les prédictions correctes
    correct_predictions = tf.equal(tf.argmax(Y, axis=1), tf.argmax(y_pred, axis=1))
    
    # Convertir les booléens en floats (0.0 pour False, 1.0 pour True)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return accuracy