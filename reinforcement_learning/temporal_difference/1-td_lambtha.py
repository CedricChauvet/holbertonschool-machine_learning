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
        state = 0  # le jouer debute en haut a gauche
        env.reset()
        z = np.zeros(env.observation_space.n)
        done = False
        truncated = False
        while not (done or truncated):
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
