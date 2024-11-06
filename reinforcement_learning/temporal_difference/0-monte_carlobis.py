#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def sample_episode(env, policy, gamma, max_steps=100):
    """
    Jouer un episode entier
    """
    SAR_list = []
    observation = 0  # le joueur débute en haut à gauche
    G = 0
    for j in range(max_steps):
        action = policy(observation)
        next_observation, reward, done, truncated, _ = env.step(action)

        # modification des reward en cas de niveau non terminé
        if (done or truncated) and reward == 0:
            reward = -1

        G = reward + gamma * G
        SAR = (observation, action, reward, G)
        SAR_list.append(SAR)

        observation = next_observation

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

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Utilise  Monte carlo pour jouer a frozen lake
    """
    for i in range(episodes):
        observation = 0  # le joueur débute en haut à gauche
        env.reset()
        SAR_list = sample_episode(env, policy, gamma, max_steps)

        for s, a, r, G in SAR_list:
            V[s] = V[s] + alpha * (G - V[s])

        # Calcul de la moyenne des valeurs d'état
        V_mean = np.mean(list(V))
        print(f"Épisode {i}: Valeur moyenne des états = {V_mean:.3f} Gt{i} = {G[i]}")

    return V