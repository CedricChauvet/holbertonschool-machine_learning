import numpy as np
import gymnasium as gym
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

class BreakoutProcessor:
    """Classe pour prétraiter les observations de Breakout."""
    
    def process_observation(self, observation):
        """Normalise les observations (0-255 -> 0-1)."""
        processed = np.squeeze(observation)
        processed = processed.mean(axis=-1)
        return processed.astype(np.float32) / 255.0  # Normalise

    def process_state_batch(self, batch):
        """Traite un lot d'états (non utilisé ici, mais requis par keras-rl)."""
        return batch

class BreakoutWrapper(gym.Wrapper):
    """Wrapper pour adapter l'environnement Breakout à notre agent."""
    
    def __init__(self, env):
        super().__init__(env)
        self.processor = BreakoutProcessor()

    def reset(self, **kwargs):
        """Réinitialise l'environnement et prétraite l'observation initiale."""
        observation, info = self.env.reset(**kwargs)
        return self.processor.process_observation(observation), info

    def step(self, action):
        """Effectue une action dans l'environnement et prétraite la nouvelle observation."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        processed_observation = self.processor.process_observation(observation)
        done = terminated or truncated
        return processed_observation, reward, done, info

def create_dqn_agent(model, nb_actions):
    """Crée et configure l'agent DQN."""
    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = GreedyQPolicy()  # Utilise une politique gloutonne pour l'évaluation
    return DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy)

def play_breakout(episodes=5, render=True):
    """Joue plusieurs épisodes de Breakout avec le modèle chargé."""
    
    # Création et configuration de l'environnement
    env = gym.make('ALE/Breakout-v5', render_mode='human' if render else None)
    env = BreakoutWrapper(env)
    nb_actions = env.action_space.n

    # Chargement du modèle et création de l'agent
    model = load_model('policy.h5')
    dqn = create_dqn_agent(model, nb_actions)
    dqn.compile(optimizer=Adam(lr=1e-3))  # Compilation de l'agent (nécessaire même sans entraînement)

    scores = []
    for episode in range(episodes):
        observation, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            action = dqn.forward(observation)  # Sélection de l'action
            observation, reward, done, _ = env.step(action)
            score += reward
            env.render()
        scores.append(score)
        print(f"Episode {episode + 1}: Score = {score}")

    env.close()
    print(f"\nAverage Score over {episodes} episodes: {np.mean(scores):.2f}")

if __name__ == "__main__":
    # Point d'entrée principal du script
    play_breakout()
