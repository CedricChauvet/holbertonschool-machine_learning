#!/usr/bin/env python3
"""
Train an agent to play atari game
"""

import gymnasium as gym
import time

# Créer l'environnement Breakout
env = gym.make("ALE/CartPole-v0", render_mode="human")
observation, info = env.reset()

# Jouer quelques actions pour voir le jeu
for _ in range(1000):
    # Choisir une action aléatoire (0-3)
    action = env.action_space.sample()
    
    # Effectuer l'action et obtenir les résultats
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Petit délai pour mieux voir le jeu
    time.sleep(0.05)
    
    # Si l'épisode est terminé, recommencer
    if terminated or truncated:
        observation, info = env.reset()

env.close()