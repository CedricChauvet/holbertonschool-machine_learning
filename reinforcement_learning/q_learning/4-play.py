#!/usr/bin/env python3
"""
Project Q learning
by Ced
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    play a full game
    """
    state = env.reset()
    state = state[0]

    for step in range(max_steps):
        action = np.argmax(Q[state])
        print (action)        
        new_state, reward, done, _, _ = env.step(action)
        state = new_state
        if done:
            if reward == 1:
                print(reward)
                break
            if reward == 0:
                print(fall)
                break
    return reward
