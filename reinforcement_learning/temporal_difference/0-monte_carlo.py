#!/usr/bin/env python3
import numpy as np


def monte_carlo(env, V, policy, episodes=5,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Utilise  Monte carlo pour jouer a frozen lake
    """

    for i in range(episodes):
        observation = 0  # le jouer debute en haut a gauche
        reward = 0
        env.reset()
        episode_list = []
        Gt = 0
        # Jouer un episode entier
        for j in range(max_steps):
            action = policy(observation)
            observation, reward, done, truncated, _ = env.step(action)

            # modification des reward en cas de niveau non terminé
            if (done or truncated) and reward == 0:
                reward = -1

            SAR = (observation, action, reward)
            episode_list.append(SAR)

            if done or truncated:
                break

        """
        Mise a jour de la valeur de l'etat pour chaque etat visite
        """
        Gt = 0

        for k in range(len(episode_list), 0, -1):
            observation, action, reward = episode_list[k-1]
            Gt = reward + gamma * Gt
            print("k", k-1, "gt", Gt)
            V[observation] = V[observation] + alpha * (Gt - V[observation])

    return V
