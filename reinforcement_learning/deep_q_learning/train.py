# Forcer l'import de Keras depuis TensorFlow
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten

from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import gymnasium as gym

# create the environment
env = gym.make("ALE/Breakout-v5", obs_type="grayscale")

nb_actions = env.action_space.n
space = env.observation_space

# affiche les dimansion de l'écran
print(space)
print(space.shape)

window_length =  space.shape[0] * space.shape[1]

# Empilement des frames (important pour les réseaux de neurones)
# env = FrameStack(env, num_stack=4)    !!!! voir si on garde cette portion de code



# # construct a MLP
model = Sequential()
model.add(Flatten(input_shape=(window_length)))
model.add(Dense(64, activation='relu'))  # hidden layer 1
model.add(Dense(64, activation='relu'))  # hidden layer 2
model.add(Dense(nb_actions, activation='linear'))  # output layer



# Adaptation des méthodes pour compatibilité keras-rl
def reset(self):
    obs, info = self.env.reset()
    return obs

def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    return obs, reward, terminated or truncated, info

def render(self):
    return self.env.render()

# Surcharger ces méthodes si nécessaire
gym.Env.reset = reset
gym.Env.step = step
gym.Env.render = render
