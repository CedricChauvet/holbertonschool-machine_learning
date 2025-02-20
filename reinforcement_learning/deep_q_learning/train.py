# Forcer l'import de Keras depuis TensorFlow
import numpy as np
import tensorflow as tf
import gymnasium as gym

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, Input, Dropout, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback

# Gestion explicite du GPU
def configure_gpu():
    # Liste des GPU disponibles
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Sélection du premier GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # Configuration mémoire GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Limitation de la mémoire GPU si nécessaire
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[0],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            # )
            
            print(f"GPU configuré : {gpus[0].name}")
            return True
        
        except RuntimeError as e:
            print("Erreur de configuration GPU :", e)
            return False
    else:
        print("Aucun GPU disponible. Utilisation du CPU.")
        return False

with tf.device('/GPU:0' if configure_gpu() else '/CPU:0'):

    class BreakoutWrapper(gym.Wrapper):
        """Wrapper pour adapter l'environnement Breakout à keras-rl"""
        def __init__(self, env):
            super().__init__(env)
            self.env = env

        def reset(self, **kwargs):
            obs, _ = self.env.reset(**kwargs)

            # print("obs type", type(obs.dtype))   # <class 'numpy.uint8'>
            return obs

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            return obs, reward, done, info


    # Création de l'environnement avec le wrapper
    env = gym.make('ALE/Breakout-v5',
                obs_type='grayscale', frameskip=4)

    nb_actions = env.action_space.n

    # Wrapper to fit with fit method
    env = BreakoutWrapper(env)

    model = Sequential([
            # Couche d'entrée 4frames, 210x160 from the environment
            Input(shape=(4, 210, 160)),

            # permuting the dimensions of the input tensor
            tf.keras.layers.Permute((2, 3, 1)),

            # resizing the input tensor to 84x84 in order to reduce the computation costs
            tf.keras.layers.Resizing(
                height=84,
                width=84,
                interpolation='bilinear'
            ),

            # Couche de rescaling
            Rescaling(1./255.0),

            # Première couche convolutionnelle avec restructuration des données
            Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                padding='same', input_shape=(84, 84, 4)),

            # Autres couches convolutionnelles
            Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'),
            Flatten(),

            Dense(512, activation='relu'),
            Dropout(0.2),
            Dense(nb_actions, activation='softmax')
        ])
    model.compile(optimizer=Adam(learning_rate=0.00025), loss='mse')

    # Configuration de la mémoire et de la politique
    memory = SequentialMemory(limit=250000, window_length=4)

    #policy =  EpsGreedyQPolicy(eps=0.1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                nb_steps=1000000)
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
                print(f"Step {self.step_count}: Reward: {self.total_reward}")
                self.total_reward = 0  # Réinitialiser pour la prochaine période


    # Créez une instance de votre callback personnalisé
    reward_logger = RewardLogger(log_interval=500)

    # Ajoutez-le à la liste des callbacks existants ou créez une nouvelle liste
    callbacks = [reward_logger]  # Ajoutez

    dqn.fit(env, nb_steps=5000000, callbacks=callbacks, visualize=False, verbose=0)
    dqn.save_weights('policyGPU.h5', overwrite=True)

    env.close()
    print("\nEntraînement terminé")
    # dqn.model.save('policy2.h5')
    # dqn.save_weights('policy.h5')
