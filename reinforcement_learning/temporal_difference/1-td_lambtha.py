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
    env.reset()
    
    for j in range(max_steps):
        # Augmenter la trace pour l'état actuel
        eligibility[observation] += 1
        
        # Choisir une action aléatoire (politique exploratrice)
        action = policy(observation)
        
        # Observer la transition
        next_obs, reward, done, truncated, _ = env.step(action)
        


        
        # Calculer TD target
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * V[next_obs]
        
        
        td_error = td_target - V[observation]
        
        
        # episode_td_errors.append(abs(td_error))
        
        # Mettre à jour la fonction de valeur pour TOUS les états
        # selon leurs traces d'éligibilité
        self.V += self.alpha * td_error * eligibility
        
        # Mettre à jour les traces d'éligibilité
        eligibility *= gamma * lambda
        
        observation = next_obs
        if truncated:
            break

def calculate_returns(episode, gamma):
    """
    retroaction des scores
    apres avoir joué un episode, on calcule les retours
    """
    returns = []
    G = 0
    for _, _, reward in reversed(episode):
        G = reward + gamma * G
        G = round(G, 3)
        returns.insert(0, G)
    if returns[-1] == 1:
        print("returns", returns)
    return returns

def td_lambtha(env, V, policy, lambtha, episodes=5000,
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
