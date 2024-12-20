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
    render = []

    for step in range(max_steps):
        graph = env.render()
        render.append(graph)
        # print(graph)
        action = np.argmax(Q[state])
        # print (action)
        new_state, reward, done, _, _ = env.step(action)
        state = new_state
        if done:
            if reward == 1:
                # print("win")
                break
            if reward == 0:
                # print("fall")
                break

    graph = env.render()
    render.append(graph)
    env.close()  # close the environment

    return reward, render
