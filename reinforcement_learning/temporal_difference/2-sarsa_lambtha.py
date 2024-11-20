#!/usr/bin/env python3
"""
Exercise with the Sarsa(λ) algorithm and the FrozenLakeEnv environment
By Ced
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    run 5000 episodes of sarsa(λ) algorithm
    """

    # Initialize eligibility traces, Q is given
    n_states, n_actions = Q.shape
    E = np.zeros((n_states, n_actions))

    for episode in range(episodes):
        """
        reset the environment and sample one episode
        player start upperleft
        Q is given
        """

        E.fill(0)  # Reset eligibility traces
        done = False
        truncated = False

        # initialize state action
        state = env.reset()[0] # this gives  0
        action = get_action(state, Q, epsilon)
        i = 0 # steps
        while not (done or truncated):
            # observing next state and next action
            next_state, reward, done, truncated, _ = env.step(action)
            if i == max_steps:
                truncated = True
    
            if done or truncated:
                next_action = None
            else:
                next_action = get_action(next_state, Q, epsilon)

            # SARSA update
            if next_action is not None:
                target = reward + gamma * Q[next_state, next_action]
            else:
                # terminating reward = O if fall, 1 if win
                target = reward

            delta = target - Q[state, action]

            # Update eligibility trace for the current state
            # and Q values
    
            E[state, action] += 1  # Update eligibility
            Q += alpha * delta * E  # update Qvalue
            E *= gamma * lambtha

            # update state and action
            state, action = next_state, next_action

            if done or truncated:   
                break
            
            # increment steps counter
            i += 1

        # Decay epsilon after each episode
        max_epsilon = 1
        
        exp = np.exp(-epsilon_decay * episode)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * exp    
    return Q


def get_action(state, Q, epsilon):
    """
    Choose action using epsilon-greedy policy
    """
    n_actions = Q.shape[1]
    if np.random.random() < epsilon:
        return np.random.choice(n_actions)
    return np.argmax(Q[state])
