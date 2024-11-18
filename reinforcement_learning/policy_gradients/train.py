#!/usr/bin/env python3
"""
This project is about policy gradient
By Ced
"""
import numpy as np
import gymnasium as gym
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Task 3, training the cartpole agent
    displays the score every episode
    render human every 1000 episodes
    """

    # scores array
    scores = []

    # get the number of states and actions from the environment gymnasium
    n_states, n_actions = env.observation_space.shape[0], env.action_space.n

    weights = np.random.rand(n_states, n_actions)

    for i in range(nb_episodes):

        # set done to False a initial state
        done = False

        # display an episode every 1000 episodes
        # if show_result is set to True
        if show_result and i % 1000 == 0:
            env = gym.make('CartPole-v1', render_mode="human")
        else:
            env = gym.make('CartPole-v1', render_mode=None)

        state, _ = env.reset()
        rewards = []
        gradients = []

        while not done:
            # get the action and gradient
            action, grad = policy_gradient(state, weights)
            state, reward, done, truncated, _ = env.step(action)

            # apply the policy gradient
            gradients.append(grad)
            rewards.append(reward)

            weights += alpha * sum([grad * (gamma ** t) * reward for t,
                                   (grad, reward) in enumerate(
                                    zip(gradients, rewards))])

            # quit after 500 steps
            if done:
                break

        # close each episode, close the window
        env.close()
        scores.append(sum(rewards))

        # display the score every episode
        print("EP: " + str(i) + " Score: " + str(sum(rewards)))

    return scores
