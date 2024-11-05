#!/usr/bin/env python3
import numpy as np


def sample_episode(env, policy, max_steps=100):
    """ Jouer un episode entier
    """
    
    SAR_list = []
    observation = 0  # le jouer debute en haut a gauche
    for j in range(max_steps):
        action = policy(observation)
        observation, reward, done, truncated, _ = env.step(action)

        # modification des reward en cas de niveau non terminé
        if (done or truncated) and reward == 0:
            reward = -1

        SAR = (observation, action, reward)
        SAR_list.append(SAR)

        if done or truncated:
            break

    return SAR_list

def calculate_returns(episode, gamma):
        """
        retroaction des scores
        """
        returns = []
        G = 0
        for _, _, reward in reversed(episode):
            G = reward + gamma * G
            returns.insert(0, G)
        print("returns", returns)
        return returns        

def update_value_function(V, episode, returns, alpha=0.1):
    """
    Met à jour la fonction de valeur V en utilisant un épisode et ses retours.
    
    Args:
    V (dict): La fonction de valeur actuelle, un dictionnaire avec les états comme clés.
    episode (list): Liste de tuples (état, action, récompense) pour un épisode.
    returns (list): Liste des retours calculés pour chaque étape de l'épisode.
    alpha (float): Taux d'apprentissage (learning rate).
    
    Returns:
    dict: La fonction de valeur V mise à jour.
    """
    visited_states = set()
    
    for (state, _, _), G in zip(episode, returns):
        if state not in visited_states:
            if state not in V:
                V[state] = 0  # Initialisation si l'état n'est pas encore dans V
            
            # Mise à jour incrémentale de V(s)
            V[state] += alpha * (G - V[state])
            visited_states.add(state)
    
    return V


def monte_carlo(env, V, policy, episodes=5,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Utilise  Monte carlo pour jouer a frozen lake
    """
    for _ in range(episodes):
        episode = sample_episode(env, policy, max_steps)
        returns = calculate_returns(episode, gamma)
        V = update_value_function(V, episode, returns, alpha)
    
    return V