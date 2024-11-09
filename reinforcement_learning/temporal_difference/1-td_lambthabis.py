#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):

    for episode in range(episodes):
        # Réinitialiser les traces d'éligibilité au début de chaque épisode
        eligibility = np.zeros(env.observation_space.n)
        
        state, _ = env.reset()
        done = False
        episode_td_errors = []
        
        while not done:
            # Augmenter la trace pour l'état actuel
            eligibility[state] += 1
            
            # Choisir une action aléatoire (politique exploratrice)
            action = policy(state)
            
            # Observer la transition
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Calculer l'erreur TD
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * V[next_state]
            td_error = td_target - V[state]
            episode_td_errors.append(abs(td_error))
            
            # Mettre à jour la fonction de valeur pour TOUS les états
            # selon leurs traces d'éligibilité
            V += alpha * td_error * eligibility
            
            # Mettre à jour les traces d'éligibilité
            eligibility *= gamma * lambtha
            
            state = next_state
            if truncated:
                break
    return V