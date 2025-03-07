#!/usr/bin/env python3
"""
Exercise with the TD(λ) algorithm and the FrozenLakeEnv environment
By Ced
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    run 5000 episodes of TD(λ) algorithm
    """
    for episode in range(episodes):
        # reset the environment and sample one episode
        # le jouer debute en haut a gauche
        state = env.reset()[0]
        z = np.zeros(env.observation_space.n)
        done = False
        truncated = False
        steps=0
        while not (done or truncated) and steps < max_steps:
            steps += 1
            action = policy(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Calcul de l'erreur TD
            td_error = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace for the current state
            z[state] += 1

            # Update each state's value and eligibility trace
            V += alpha * td_error * z

            # Apply lambtha decay to eligibility traces
            z *= gamma * lambtha

            state = next_state

    return V
# "end"
