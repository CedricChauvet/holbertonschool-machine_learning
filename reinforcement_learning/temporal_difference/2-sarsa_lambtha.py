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
        
        n_states, n_actions = Q.shape
        action =  get_action(state, Q, epsilon)

        # Initialize Q-table and eligibility traces
        E = np.zeros((n_states, n_actions))

        done = False
        truncated = False

        while not (done or truncated):
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Choose next action according to epsilon-greedy policy

            next_action = get_action(next_state, Q, epsilon)

            # Calculate TD error
            td_error = (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            # Update eligibility trace for current state-action pair
            E[state, action] += 1
            
            # Update Q-values for all state-action pairs
            Q += alpha * td_error * E
            
            # Decay eligibility traces
            E *= gamma * lambtha

            # Move to next state-action pair
            state = next_state
            action = next_action
            
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