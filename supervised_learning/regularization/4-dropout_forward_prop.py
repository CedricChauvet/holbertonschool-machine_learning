#!/usr/bin/env python3
"""
Regularization project
by Ced
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    """
    
    
    cache = dict()
    for i in range(L ):
        if i == 0:
            cache['A0'] = X
        else:
            z = np.dot(weights[f"W{i}"], cache[f"A{i-1}"]
                ) + weights[f"b{i}"]
            
            tanh_z = np.tanh(z)
            cache[f"A{i}"] = tanh_z
            print(f"shape W{i}",weights[f"W{i}"].shape)
            
            x=np.random.randint(100, size=())
    
    cache[f"A{L}"] = np.dot(weights[f"W{L}"], cache[f"A{L-1}"]) + weights[f"b{L}"]
    
    
    return cache
