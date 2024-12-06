import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Charger le modèle pré-entraîné
model = load_model('cartpole_score_8000.h5')

# Créer l'environnement Cartpole
env = gym.make('CartPole-v1', render_mode='human')

# Nombre d'épisodes à jouer
num_episodes = 10

# Boucle pour jouer les épisodes
for episode in range(num_episodes):
    # Réinitialiser l'environnement
    observation, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Convertir l'observation en tableau numpy et ajouter une dimension pour le batch
        observation_reshaped = observation.reshape(1, -1)
        
        # Prédire l'action à prendre
        action = np.argmax(model.predict(observation_reshaped, verbose=0)[0])
        
        # Effectuer l'action dans l'environnement
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Mettre à jour le total des récompenses
        total_reward += reward
        
        # Vérifier si l'épisode est terminé
        done = terminated or truncated
    
    print(f"Episode {episode + 1} - Reward: {total_reward}")

# Fermer l'environnement
env.close()