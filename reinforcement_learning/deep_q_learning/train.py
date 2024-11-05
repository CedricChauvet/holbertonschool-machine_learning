#!/usr/bin/env python3
"""
Train an agent to play atari game
"""

import gymnasium as gym
import tensorflow as tf
import time
import rl
"""
rl is a package that contains the implementation of the DQN algorithm
it contains the following modules:
- DQNAgent: the agent that will learn to play the game
- Sequential Memory: the memory that will store the experiences
- EpsGreedyPolicy: the policy that will be used to select the actions
"""


# Create the Breakout environment
env = gym.make("ALE/Breakout-v5", render_mode="human") # Breakout-v5 is the casse brique

"""
reset() : Réinitialise l'environnement et renvoie l'observation initiale.
step(action) : Exécute une action et renvoie l'observation suivante, la
récompense, un indicateur de fin d'épisode, et des informations supplémentaires.
render() : Affiche l'état actuel de l'environnement: render_mode="human")
"""
print("actions", env.action_space.n)
# l'agent
agent = rl.agent.DQNAgent(model=env, nb_actions=env.action_space.n,  #4 actions possible
                          # ....
        )



model = tf.keras.Sequential()

