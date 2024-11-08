import gymnasium as gym
import time
import numpy as np
import random

# Créer l'environnement



env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)  # Ajout du render_mode pour visualisation


# Reset retourne un tuple (observation, info)
initial_observation, info = env.reset()  # Capturer les deux valeurs retournées
print("Observation initiale:", initial_observation) 
# Définir les actions possibles
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

# Faire un pas dans l'environnement
action = 2

next_observation, reward, done, truncated, info = env.step(RIGHT)
time.sleep(1)
next_observation, reward, done, truncated, info = env.step(RIGHT)
time.sleep(1)
next_observation, reward, done, truncated, info = env.step(RIGHT)
time.sleep(1)


print("État initial:", initial_observation)
print("Nouvel état après RIGHT:", next_observation)
print("Récompense:", reward)
print("Episode terminé ?", done)

env.close()