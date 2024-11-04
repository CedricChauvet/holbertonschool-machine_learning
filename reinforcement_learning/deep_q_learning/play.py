import numpy as np
import gymnasium as gym
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory


class BreakoutWrapper(gym.Wrapper):
    """Wrapper pour adapter l'environnement Breakout à notre agent."""
    
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        """Réinitialise l'environnement et prétraite l'observation initiale."""
        observation, info = self.env.reset(**kwargs)
        processed = np.squeeze(observation) # supprime les dimensions inutiles
        processed = processed.mean(axis=-1) # Convertit en niveaux de gris
        processed = processed.astype(np.float32) / 255.0  # Normalise
        return processed

    def step(self, action):
        """Effectue une action dans l'environnement et prétraite la nouvelle observation."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        processed = np.squeeze(observation) # supprime les dimensions inutiles
        processed = processed.mean(axis=-1) # Convertit en niveaux de gris
        processed = processed.astype(np.float32) / 255.0  # Normalise
        done = terminated or truncated
        return processed, reward, done, info
    
    def render(self, *args, **kwargs):
        return self.env.render()

def play_breakout(episodes=5, render=True):
    """Joue plusieurs épisodes de Breakout avec le modèle chargé."""
    
    # Création et configuration de l'environnement
    env = gym.make('Breakout-v4', render_mode='human' if render else None)
    env = BreakoutWrapper(env)
    nb_actions = env.action_space.n

    model = Sequential([
            # Couche d'entrée 4frames, 210x160
            Input(shape=(4, 210, 160)),

            # Première couche convolutionnelle avec restructuration des données
            Conv2D(32, (8, 8), strides=(4, 4), activation='elu',
                data_format='channels_first'),

            # Autres couches convolutionnelles
            Conv2D(64, (4, 4), strides=(2, 2), activation='elu'),
            Conv2D(64, (3, 3), strides=(1, 1), activation='elu'),
            Flatten(),

            Dense(512, activation='relu'),
            Dropout(0.2),
            Dense(nb_actions, activation='linear')
        ])

    # Chargement du modèle et création de l'agent
    model.load_weights('policy.h5')

    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = GreedyQPolicy()  # Utilise une politique gloutonne pour l'évaluation

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy)
    dqn.compile(optimizer=Adam(lr=1e-3))  # Compilation de l'agent (nécessaire même sans entraînement)
    dqn.test(env, nb_episodes=2, visualize=True)
    exit()
    # scores = []
    # for episode in range(episodes):
    #     observation = env.reset()
    #     done = False
    #     score = 0
        
    #     while not done:
    #         action = dqn.forward(observation)  # Sélection de l'action
    #         observation, reward, done, _ = env.step(action)
    #         print("action" , action)
    #         score += reward
    #         env.render()
    #     scores.append(score)
    #     print(f"Episode {episode + 1}: Score = {score}")

    # env.close()
    # print(f"\nAverage Score over {episodes} episodes: {np.mean(scores):.2f}")

if __name__ == "__main__":
    # Point d'entrée principal du script
    play_breakout()
