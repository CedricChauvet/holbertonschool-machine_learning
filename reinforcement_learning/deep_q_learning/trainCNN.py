# Forcer l'import de Keras depuis TensorFlow
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import gymnasium as gym
import time
from rl.callbacks import Callback
from gymnasium import Wrapper


class BreakoutWrapper(gym.Wrapper):
    """Wrapper pour adapter l'environnement Breakout à keras-rl"""
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        # Normalisation des observations (0-255 -> 0-1)
        return np.array(obs, dtype=np.float32) / 255.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        # Normalisation des observations (0-255 -> 0-1)
        return np.array(obs, dtype=np.float32) / 255.0, reward, done, info


# Création de l'environnement avec le wrapper
env = gym.make('ALE/Breakout-v5',
               obs_type='grayscale', frameskip=4)
nb_actions = env.action_space.n
env = BreakoutWrapper(env)

model = Sequential([
        # Couche d'entrée 4frames, 210x160
        Input(shape=(4, 210, 160)),

        # Première couche convolutionnelle avec restructuration des données
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
               data_format='channels_first'),

        # Autres couches convolutionnelles
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        Flatten(),

        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(nb_actions, activation='softmax')
    ])
model.compile(optimizer=Adam(learning_rate=0.00025), loss='mse')

# Configuration de la mémoire et de la politique
memory = SequentialMemory(limit=500000, window_length=4)
policy = EpsGreedyQPolicy(eps=0.1)

# Configuration de l'agent DQN
dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    nb_steps_warmup=1000,
    target_model_update=1e-2,
    policy=policy,
    gamma=0.99,
    train_interval=4,
    delta_clip=1.
)

optimizer = Adam(lr=1e-3)
# Compilation de l'agent
dqn.compile(
    optimizer=optimizer,
    metrics=['mse']
)


# Callbacks
class RewardLogger(Callback):
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_count = 0
        self.total_reward = 0

    def on_step_end(self, step, logs={}):
        self.step_count += 1
        self.total_reward += logs.get('reward', 0)

        if self.step_count % self.log_interval == 0:
            Average_Reward = self.total_reward / self.log_interval
            print(f"Step {self.step_count}: Average_Reward: {Average_Reward}")
            self.total_reward = 0  # Réinitialiser pour la prochaine période


# Créez une instance de votre callback personnalisé
reward_logger = RewardLogger(log_interval=500)

# Ajoutez-le à la liste des callbacks existants ou créez une nouvelle liste
callbacks = [reward_logger]  # Ajoutez

dqn.fit(env, nb_steps=70000, callbacks=callbacks, visualize=False, verbose=0)

env.close()
print("\nEntraînement terminé")
# dqn.model.save('policy2.h5')
dqn.save_weights('policy.h5')

