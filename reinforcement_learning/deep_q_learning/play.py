"""
play method for the game
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Dropout, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback
import time
import gymnasium as gym
from keras.models import load_model
import numpy as np


env = gym.make("ALE/Breakout-v5", frameskip=4, render_mode="human", obs_type="rgb")

# Configuration du modèle et de l'agent
input_shape = (4, 210, 160)
nb_actions = env.action_space.n

model = Sequential([
        Input(shape=input_shape),
        tf.keras.layers.Permute((2, 3, 1)),
        tf.keras.layers.Resizing(84, 84, interpolation='bilinear'),
        Rescaling(1./255.0),
        
        # Architecture plus profonde
        Conv2D(32, (8, 8), strides=4, activation='relu', padding='valid'),
        Conv2D(64, (4, 4), strides=2, activation='relu', padding='valid'),
        Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid'),
        Conv2D(128, (3, 3), strides=1, activation='relu', padding='valid'),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(nb_actions, activation='linear')  # Linear au lieu de softmax pour Q-learning
    ])



# Fonctions utilitaires
def process_observation(obs):
    # S'assurer que l'observation a la bonne forme (210, 160)
    if obs.shape != (210, 160, 3):
        raise ValueError(f"Forme d'observation inattendue: {obs.shape}")
    
    # Convertir en niveaux de gris si l'observation est en couleur
    gray_obs = np.mean(obs, axis=2).astype(np.uint8)
    return gray_obs

def update_frame_buffer(buffer, new_frame):
    # Décaler les frames et ajouter la nouvelle
    buffer[:-1] = buffer[1:]
    buffer[-1] = new_frame
    return buffer

model.load_weights('policyGPU_better.h5')
observation, info = env.reset()
# Pour stocker les 4 dernières frames
frame_buffer = np.zeros((4, 210, 160), dtype=np.uint8)
frame_buffer[-1] = process_observation(observation)

episode_over = False
total_reward = 0
while not episode_over:
      # Préparer l'entrée pour le modèle (batch de taille 1)
    model_input = np.expand_dims(frame_buffer, axis=0)
    
    # Obtenir l'action du modèle
    q_values = model.predict(model_input, verbose=0)[0]
    action = np.argmax(q_values)
    
    # Exécuter l'action
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Mettre à jour le buffer de frames
    frame_buffer = update_frame_buffer(frame_buffer, process_observation(observation))
    
    total_reward += reward
    time.sleep(0.05)  # Ralentir légèrement pour mieux visualiser

    episode_over = terminated or truncated


print(f"Partie terminée! Score total: {total_reward}")
env.close()