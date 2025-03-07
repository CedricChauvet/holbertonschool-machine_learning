"""
new version of sarsa
"""
import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, initial_epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Implémentation de l'algorithme Sarsa(λ) pour l'apprentissage par renforcement
    """
    
    # Initialisation
    n_states, n_actions = Q.shape
    epsilon = initial_epsilon
    
    for episode in range(episodes):
        # Réinitialiser les traces d'éligibilité au début de chaque épisode
        E = np.zeros((n_states, n_actions))
        
        # Initialiser l'état et l'action
        state = env.reset()[0]
        action = get_action(state, Q, epsilon)
        
        # Compteur de pas pour l'épisode actuel
        steps = 0
        done = truncated = False
        
        while not (done or truncated):
            # Incrémenter le compteur de pas
            steps += 1
            
            # Vérifier si le nombre maximum de pas est atteint
            if steps >= max_steps:
                truncated = True
                
            # Exécuter l'action et observer le nouvel état et la récompense
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Sélectionner la prochaine action selon la politique epsilon-greedy
            if not (done or truncated):
                next_action = get_action(next_state, Q, epsilon)
            else:
                next_action = None
                
            # Calculer la cible pour la mise à jour Q
            if next_action is not None:
                target = reward + gamma * Q[next_state, next_action]
            else:
                target = reward
                
            # Calculer l'erreur TD
            delta = target - Q[state, action]
            
            # Mettre à jour la trace d'éligibilité pour l'état-action actuel
            E[state, action] += 1
            
            # Mise à jour de toutes les valeurs Q selon les traces d'éligibilité
            Q += alpha * delta * E
            
            # Décroissance des traces d'éligibilité
            E *= gamma * lambtha
            
            # Transition vers le nouvel état et la nouvelle action
            if not (done or truncated):
                state, action = next_state, next_action
            else:
                break
    # Exploration rate decay
            epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                    np.exp(-epsilon_decay * episode))    
            i += 1
            
        
    return Q

def get_action(state, Q, epsilon):
    """
    Choose action using epsilon-greedy policy
    """
    n_actions = Q.shape[1]  
    if np.random.random() < epsilon:
        return np.random.choice(n_actions)
    return np.argmax(Q[state,:])
