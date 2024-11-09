#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    
    
    
    for episode in range(episodes):
        # reset the environment and sample one episode
        SAR_list = []
        obs = 0  # le jouer debute en haut a gauche
        env.reset()
        eligibility = np.zeros(env.observation_space.n)
        done = False
        while  not done:
            # Augmenter la trace pour l'état actuel
            
            # Choisir une action aléatoire (politique exploratrice)
            action = policy(obs)
            
            # Observer la transition
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # calculer TD target  
            if done:
                td_target = reward
            # calcul de target avec discount factor
            else:
                td_target = reward + gamma * V[next_obs]    
            
            # on deroule l'erreur TD
            td_error = td_target - V[obs]

            V += alpha * td_error * eligibility 
            
            
            eligibility *= gamma * lambtha 
            eligibility[obs] += 1


            obs = next_obs
    
            if truncated:
                break
    
    return V