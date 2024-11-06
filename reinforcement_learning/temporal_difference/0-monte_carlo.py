#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def sample_episode(env, policy, max_steps=10000):
    """
    Jouer un episode entier
    """
    global count_victory
    SAR_list = []
   
    env.reset()
    observation = 0  # le jouer debute en haut a gauche
    for j in range(max_steps):
        
        action = policy(observation)

        observation, reward, done, truncated, _ = env.step(action)

        # modification des reward en cas de niveau non terminé
        # if done and reward == 0:
        #     reward = -1

        # i m trully sorry for this, but it's the only way to get the output
        if env.unwrapped.desc[observation // 8, observation % 8] == b'H':
            print("fall") 

        if (truncated):
            print("truncated")   # see if it's happening
        # State Action Reward
        SAR = (observation, action, reward)
        SAR_list.append(SAR)

        if reward == 1:
            count_victory += 1
        # si le niveau est terminé ou tronqué, on s'arrête
        if done or truncated:
            break
    env.close()
    return SAR_list


def calculate_returns(list_one_episode, gamma):
    """
    retroaction des scores
    apres avoir joué un episode, on calcule les retours
    """
    Gt = []
    G = 0

    for _, _, reward in reversed(list_one_episode):
        G = reward + gamma * G
        G = round(G,3) # rounded for esthetic reasons
        Gt.append(G)    
     
    if Gt[0] == 1:
        print("victoire",(Gt[::-1]),"\n\n")
    return Gt[::-1]


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

    print(count_victory, "Vicories over", episodes, "episodes")
    return V
