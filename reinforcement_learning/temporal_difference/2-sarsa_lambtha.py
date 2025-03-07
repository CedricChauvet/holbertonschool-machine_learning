#!/usr/bin/env python3
"""
Improved version of SARSA(λ)
"""
import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                 gamma=0.99, initial_epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ) reinforcement learning algorithm.
    """
    # Validation
    if not (0 <= lambtha <= 1):
        raise ValueError("lambtha must be between 0 and 1")
        
    n_states, n_actions = Q.shape
    epsilon = initial_epsilon
    E = np.zeros((n_states, n_actions))
    
    for episode in range(episodes):
        E.fill(0)
        state = env.reset()[0]
        action = get_action(state, Q, epsilon)
        
        steps = 0
        done = truncated = False
        
        while not (done or truncated):
            steps += 1
            if steps >= max_steps:
                truncated = True
                
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = get_action(next_state, Q, epsilon) if not (done or truncated) else 0
            
            # Calcul optimisé
            delta = reward + (gamma * Q[next_state, next_action] if not (done or truncated) else 0) - Q[state, action]
            update_factor = alpha * delta
            decay_factor = gamma * lambtha
            
            E[state, action] += 1
            Q += update_factor * E
            E *= decay_factor
            
            if not (done or truncated):
                state, action = next_state, next_action
            else:
                break
        
        # Décroissance linéaire corrigée
        epsilon = max(min_epsilon, initial_epsilon - (initial_epsilon - min_epsilon) * (episode / episodes))
    
    return Q

def get_action(state, Q, epsilon):
    """Choose action using epsilon-greedy policy"""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, Q.shape[1])
    return np.argmax(Q[state, :])