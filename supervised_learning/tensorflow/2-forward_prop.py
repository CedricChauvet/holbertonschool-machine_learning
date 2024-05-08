#!/usr/bin/env python3
"""
2 task on tensoflow project
"""
import tensorflow.compat.v1 as tf


create_placeholders = __import__('0-create_placeholders').create_placeholders
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Forward Propagation
    """

    prev = x
    for n, activation in zip(layer_sizes, activations):
        lay = create_layer(prev, n, activation)
        prev = lay
    return lay
