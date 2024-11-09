#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    for episode in range(episodes):
        # reset the environment and sample one episode
        state = 0  # le jouer debute en haut a gauche
        env.reset()
        z = np.zeros(env.observation_space.n)
        done = False
        while  not done:
            action = policy(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Calcul de la valeur courante
            current_value = V[state]
            
            # Calcul de la valeur suivante (0 si épisode terminé)
            next_value =  V[next_state]
            
            # Calcul de l'erreur TD
            td_error = reward + gamma * next_value - current_value
            
                   # Update eligibility trace for the current state
            z[state] += 1

            # Update each state's value and eligibility trace
            V += alpha * td_error * z

            # Apply lambtha decay to eligibility traces
            z *= gamma * lambtha
            
            state = next_state
            if done:
                break
         
    return V

                