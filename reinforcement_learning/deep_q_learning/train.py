# Forcer l'import de Keras depuis TensorFlow
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam  # Utiliser l'optimiseur legacy
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
        # return obs
        return np.array(obs, dtype=np.float32) / 255.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        # Normalisation des observations (0-255 -> 0-1)
        # return obs, reward, done, info
        return np.array(obs, dtype=np.float32) / 255.0, reward, done, info

# Création de l'environnement avec le wrapper
env = gym.make('ALE/Breakout-v5', render_mode='human', obs_type='rgb', frameskip=1)
nb_actions = env.action_space.n
env = BreakoutWrapper(env)

# Configuration des dimensions

input_shape =  (1, 210, 160, 3)# (1,) + env.observation_space.shape  #  pour le window_length
print(f"Input shape: {input_shape}")
print(f"Nombre d'actions: {nb_actions}")
# Construction du modèle
model = Sequential([
    Flatten(input_shape=input_shape),
    Dense(256, activation='elu'),
    Dense(128, activation='elu'),
    Dense(nb_actions, activation='linear')
])

# Configuration de la mémoire et de la politique
memory = SequentialMemory(limit=200000, window_length=1)  # Notez le window_length=1
policy = EpsGreedyQPolicy(eps=0.2)

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
# loss_fn = tf.keras.losses.mean_squared_error
# Compilation de l'agent
dqn.compile(
    optimizer=optimizer,
    metrics=['mae']
)
# def custom_fit(agent, env, nb_steps, training=True):
#     observation = env.reset()
#     done = False
#     episode = 0
#     step = 0
#     episode_reward = 0
#     episode_step = 0
    
#     while step < nb_steps:
#         if done:
#             print(f"\nÉpisode {episode} terminé - Récompense totale: {episode_reward}")
#             observation = env.reset()
#             episode_step = 0
#             episode_reward = 0
#             episode += 1
#             done = False
        
#         # Sélection et exécution de l'action
#         action = dqn.forward(observation)
#         print("action", action)
#         next_observation, reward, done, info = env.step(action)
        
#         # Apprentissage
#         if training:
#             dqn.backward(reward, terminal=done)
            
#         # Mise à jour des métriques
#         episode_reward += reward
#         episode_step += 1
#         step += 1
        
#         # Affichage périodique
#         if step % 100 == 0:
#             print(f"\rÉtape: {step}/{nb_steps} | "
#                   f"Épisode: {episode} | "
#                   f"Récompense: {episode_reward:.2f}", end="")
        
#         observation = next_observation
#         env.render()
#         time.sleep(0.05)


# # Entraînement
# custom_fit(dqn, env, nb_steps=200, training=True)

#Callbacks
class RewardLogger(Callback):
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_count = 0
        self.total_reward = 0

    def on_step_end(self, step, logs={}):
        self.step_count += 1
        self.total_reward += logs.get('reward', 0)

        if self.step_count % self.log_interval == 0:
            print(f"Step {self.step_count}: Average Reward = {self.total_reward / self.log_interval}")
            self.total_reward = 0  # Réinitialiser pour la prochaine période

# Créez une instance de votre callback personnalisé
reward_logger = RewardLogger(log_interval=500)

# Ajoutez-le à la liste des callbacks existants ou créez une nouvelle liste
callbacks = [reward_logger]  # Ajoutez

dqn.fit(env, nb_steps=8000, visualize=False, verbose=0,)

env.close()
print("\nEntraînement terminé")
model.save('breakout_model_8000')  # Sauvegarde le modèle complet
