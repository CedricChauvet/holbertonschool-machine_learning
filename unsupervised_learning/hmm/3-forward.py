#!/usr/bin/env python3
"""
Project Hiden Markov Model
By Ced+
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    n = len(Observation)
    num_S = Transition.shape[0]

    a = np.empty([num_S, n], dtype='float')
    # Base case
    print("EM", Initial[:,0].shape)
    a[:,0] = np.multiply(Initial[:,0], Emission[:, Observation[0]])

        # Recursive case
    for t in range(1, n):
        a[:,t] = np.multiply(Emission[:,Observation[t]], np.dot(Transition.T, a[:,t-1]))
        
    return (np.sum(a[:,n-1]), a)