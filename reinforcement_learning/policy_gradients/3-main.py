#!/usr/bin/env python3

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
from train_dnn import agent



if __name__ == '__main__':
    agent = agent(ALPHA=0.0005, input_dims=4, GAMMA=0.99, n_actions=2,
                    layer1_size=512, layer2_size=512, filename='cartpole_score_8000.h5')
    env = gym.make('CartPole-v1')

    score_history = []
    num_episodes = 1000

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()[0]

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done,truncated, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
            if score > 8000:
                print('score > 400 model saved')
                agent.save_model()
                exit()
                done = True

        score_history.append(score)
        env.close()
        agent.learn()

        print('episode', i, 'score %.1f' % score)