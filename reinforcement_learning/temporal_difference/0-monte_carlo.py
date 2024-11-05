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
        SAR_list.append((observation, action, 0)) # on ajoute l'etat initial, OK?
        observation, reward, done, truncated, _ = env.step(action)

        # modification des reward en cas de niveau non terminé
        if (done or truncated) and reward == 0:
            reward = -1

        # State Action Reward
        SAR = (observation, action, reward)
        SAR_list.append(SAR)

        # si le niveau est terminé ou tronqué, on s'arrête
        if done or truncated:
            break


    return SAR_list


def calculate_returns(episode, gamma):
    """
    retroaction des scores
    apres avoir joué un episode, on calcule les retours
    """
    returns = []
    G = 0
    for _, _, reward in reversed(episode):
        G = reward + gamma * G
        returns.insert(0, G)
    # print("returns", returns)
    returns[0] = 0
    return returns


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Utilise  Monte carlo pour jouer a frozen lake
    """

    for i in range(episodes):
        observation = 0  # le joueur debute en haut a gauche
        reward = 0
        env.reset()
        episode_list = []
        Gt = 0

        SAR_list = sample_episode(env, policy, max_steps)
        Gt = calculate_returns(SAR_list, gamma)

        len_list = len(SAR_list)
        len_Gt = len(Gt)

        if len_list != len_Gt:
            raise ValueError("SAR_list and Gt must have the same length")

        for k in range(len_Gt):
            s = SAR_list[k][0]
            # print("k", k, "s", s, "Gt", Gt[k])
            V[s] = V[s] + alpha * (Gt[k] - V[s])

    return V
