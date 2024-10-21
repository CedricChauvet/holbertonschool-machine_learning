#!/usr/bin/env python3
"""
Project Q learning
by Ced
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    In this game there is no accumulation of the reward.
    it a 0 if the agent is still alive but it didn't reach his goal
    it s a 1 if the agent reach the goal
    it s a -1 if the agent fall in a hole
    """
    total_reward = []

    for episode in range(episodes):

        # Did i chose the right epsilon decay formula?
        epsilon = (min_epsilon + (epsilon - min_epsilon)
                   * np.exp(-epsilon_decay*episode))
        # Reset the environment
        state = env.reset()
        state = state[0]
        done = False

        # repeat, max_steps times, with 100 steps we cant have a reward of 0
        # but...if the environment is bigger?
        for step in range(max_steps):
            # exploration or exploitation, eplislon gourmand
            action = epsilon_greedy(Q, state, epsilon)

            # Take the action and observe the outcome state and reward
            # done if the game is finished, fall? or win?
            new_state, reward, done, _, _ = env.step(action)

            if done and reward == 0:
                # just change the reward to -1
                reward = -1
            # equivalent to optimize Q-table using Q pvalue maximzer
            Q[state][action] = (Q[state][action] + alpha
                                * (reward + gamma *
                                np.max(Q[new_state]) -
                                Q[state][action]))

            # Our state is the new state
            state = new_state
            # then break if done
            if done:
                break

        total_reward.append(reward)
    return Q, total_reward
