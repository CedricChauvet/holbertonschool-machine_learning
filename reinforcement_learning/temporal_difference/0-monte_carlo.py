#!/usr/bin/env python3
import numpy as np


def sample_episode(env, policy, max_steps=100):
    """
    Jouer un episode entier
    """

    SAR_list = []
    observation = 0  # le jouer debute en haut a gauche

    for j in range(max_steps):

        action = policy(observation)

        new_obs, reward, done, truncated, _ = env.step(action)
        SAR_list.append((observation, reward))

        if done or truncated:
            break

        observation = new_obs
    return SAR_list


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Utilise  Monte carlo pour jouer a frozen lake
    """

    # show the initial state of the game
    for episode in range(episodes):

        SAR_list = sample_episode(env, policy, max_steps)
        SAR_list = np.array(SAR_list, dtype=int)

        G = 0
        for state, reward in reversed(SAR_list):
            # return apres la fin de l'episode
            G = reward + gamma * G

            # if this is a novel state
            if state not in SAR_list[:episode, 0]:
                # Update the value function V(s)
                V[state] = V[state] + alpha * (G - V[state])

    return V
