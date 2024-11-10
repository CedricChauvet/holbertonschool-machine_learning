#!/usr/bin/env python3
"""
Exercise with the Sarsa(λ) algorithm and the FrozenLakeEnv environment
By Ced
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    run 5000 episodes of TD(λ) algorithm
    """
    for episode in range(episodes):
        # reset the environment and sample one episode
        # le jouer debute en haut a gauche
        state = env.reset()[0]
        # action = get_action(state, Q, epsilon) 
        n_states, n_actions = Q.shape

        # Initialize Q-table and eligibility traces
        E = np.zeros((n_states, n_actions))

        done = False
        truncated = False

        # state = env.reset()
        action = get_action(state, Q, epsilon)
        E.fill(0)  # Reset eligibility traces
    
        while not done:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = get_action(next_state, Q, epsilon)
            
            # SARSA update
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            E[state, action] += 1
            
            for s in range(n_states):
                for a in range(n_actions):
                    Q[s, a] += alpha * delta * E[s, a]
                    E[s, a] *= gamma * lambtha
                    
            state, action = next_state, next_action

            
        # Decay epsilon après chaque épisode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q

def get_action(state, Q, epsilon):
    """
    Choose action using epsilon-greedy policy
    """
    n_actions = Q.shape[1]
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])