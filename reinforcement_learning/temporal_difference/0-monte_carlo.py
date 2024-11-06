#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def sample_episode(env, policy, max_steps=100):
    """
    Jouer un episode entier
    """
    global count_victory
    SAR_list = []
    observation = 0  # le jouer debute en haut a gauche
    for j in range(max_steps):

        action = policy(observation)

        # on ajoute l'etat initial, OK?
        SAR_list.append((observation, action, 0))
        observation, reward, done, truncated, _ = env.step(action)

        # modification des reward en cas de niveau non terminé
        if done and reward == 0:
            reward = -1

        if (truncated):
            print("truncated")   # see if it's happening
        # State Action Reward
        SAR = (observation, action, reward)
        SAR_list.append(SAR)

        if reward == 1:
            count_victory += 1
        # si le niveau est terminé ou tronqué, on s'arrête
        if done:
            break

    return SAR_list


def calculate_returns(episode, gamma):
    """
    retroaction des scores
    apres avoir joué un episode, on calcule les retours
    """
    returns = []
    G = 0
    for _, _, reward in reversed(episode):
        G = reward + gamma * G
        returns.insert(0, G)
    # print("returns", returns)
    returns[0] = 0
    return returns


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Utilise  Monte carlo pour jouer a frozen lake
    """
    global count_victory
    count_victory= 0
    
    # show the initial state of the game
    print("Frozen lake 8x8", V.reshape((8, 8)))

    for i in range(episodes):
        env.reset()

        SAR_list = sample_episode(env, policy, max_steps)
        Gt = calculate_returns(SAR_list, gamma)

        len_list = len(SAR_list)
        len_Gt = len(Gt)

        if len_list != len_Gt:
            raise ValueError("SAR_list and Gt must have the same length")

        for k in range(len_Gt):
            s = SAR_list[k][0]
            # print("k", k, "s", s, "Gt", Gt[k])
            V[s] = V[s] + alpha * (Gt[k] - V[s])

        V_mean = np.mean(list(V))
        if i % 100 == 0:
            print(f"Épisode {i}: Valeur moyenne des états = {V_mean:.3f}")
    # s, a, v = list(zip(*SAR_list, V))
    print(count_victory, "Vicories over", episodes, "episodes")
    return V
