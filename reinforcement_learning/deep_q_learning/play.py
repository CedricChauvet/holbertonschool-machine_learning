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

import gymnasium as gym
from keras.models import load_model
import numpy as np


env = gym.make("ALE/Breakout-v5", frameskip=4, obs_type='rgb')

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

model.load_weights('policyGPU_better.h5')
observation, info = env.reset()

episode_over = False
while not episode_over:
    print("shape", observation.shape)
    print("observation", observation)
    # processed_observation = np.expand_dims(observation, axis=0)  # Add batch dimension

    # Get the action from the model
    action_probs = model.predict(observation, verbose=1)
    action = np.argmax(action_probs)  # Choose the action with highest probabilityobservation, reward, terminated, truncated, info = env.step(action)
    print("action", action)
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()