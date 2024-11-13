#!/usr/bin/env python3
"""
This project is about policy gradient
By Ced
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    θ(t+1) = θ(t) + α ∑(γ^t ∇θ log π(at|st; θ) Rt)
    """
    scores = []
    n_states, n_actions = env.observation_space.shape[0], env.action_space.n

    weights = np.random.rand(n_states, n_actions)
    # grad = np.zeros(weights.shape)

    for i in range(nb_episodes):
        state, _ = env.reset()
        done = False
        # print("state: ", state)
        rewards = []
        gradients = []
        while not done:
            action, grad = policy_gradient(state, weights)
            # print("action", action)*

            next_state, reward, done, truncated, _ = env.step(action)

            gradients.append(grad)
            rewards.append(reward)

            state = next_state
            weights += alpha * sum([grad * (gamma ** t) * reward for t,
                                   (grad, reward) in enumerate(
                                    zip(gradients, rewards))])

            if truncated:
                break

        scores.append(sum(rewards))

        print("EP: " + str(i) + " Score: " + str(sum(rewards)))
    return scores
