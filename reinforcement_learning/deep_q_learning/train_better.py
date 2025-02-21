import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Dropout, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback
import gymnasium as gym

class BreakoutWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, _ = self.env.reset(**kwargs)
            self.lives = 5
        else:
            # Perdu une vie mais pas game over
            obs, _, _, _, _ = self.env.step(1)  # Action FIRE pour relancer la balle
        self.was_real_done = False
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Ajuster la récompense
        reward = np.clip(reward, -1., 1.)  # Clipper entre -1 et 1
        
        # Gérer la fin d'épisode basée sur les vies
        lives = info.get('lives', 0)
        if lives < self.lives and lives > 0:
            # Perdu une vie
            self.was_real_done = False
            reward = -1.0  # Pénalité pour perte de vie
        elif lives == 0:
            self.was_real_done = True
        self.lives = lives
        
        done = self.was_real_done or terminated or truncated
        return obs, reward, done, info

def build_model(input_shape, nb_actions):
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
    return model

# Création et configuration de l'environnement
env = gym.make('ALE/Breakout-v5', obs_type='grayscale', frameskip=1)
env = BreakoutWrapper(env)

# Configuration du modèle et de l'agent
input_shape = (4, 210, 160)
nb_actions = env.action_space.n

model = build_model(input_shape, nb_actions)

"""
trop de mémoire utilisée peut entraîner un crash du programme
Pour Breakout spécifiquement, 250 000 est un minimum acceptable
"""
memory = SequentialMemory(limit=250000, window_length=4)  # Augmenté la taille de mémoire

policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr='eps',
    value_max=1.0,
    value_min=0.1,
    value_test=0.05,
    nb_steps=1000000
)

dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    nb_steps_warmup=50000,  # Augmenté
    target_model_update=10000,  # Mis à jour moins fréquemment
    policy=policy,
    gamma=0.99,
    train_interval=4,
    delta_clip=1.,
    batch_size=32,
    enable_double_dqn=True  # Activé Double DQN
)

dqn.compile(Adam(learning_rate=0.00025, clipnorm=1.0))

# Callback amélioré pour le suivi
class EnhancedRewardLogger(Callback):
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0
        self.rewards_history = []
        self.episode_rewards = 0

    def on_episode_end(self, episode, logs={}):
        self.episode_count += 1
        self.rewards_history.append(self.episode_rewards)
        self.episode_rewards = 0
        
        if self.episode_count % 10 == 0:
            avg_reward = np.mean(self.rewards_history[-10:])
            print(f"Episode {self.episode_count}: Average Reward (last 10): {avg_reward:.2f}")

    def on_step_end(self, step, logs={}):
        reward = logs.get('reward', 0)
        self.step_count += 1
        self.total_reward += reward
        self.episode_rewards += reward

        if self.step_count % self.log_interval == 0:
            print(f"Step {self.step_count}: Total Reward: {self.total_reward}")
            self.total_reward = 0

callbacks = [EnhancedRewardLogger(log_interval=1000)]


dqn.fit(env, nb_steps=1000000, callbacks=callbacks, visualize=False, verbose=0)
dqn.save_weights('policyGPU.h5', overwrite=True)

env.close()
print("\nEntraînement terminé")
# dqn.model.save('policy2.h5')