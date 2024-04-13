#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))
    colors = ["red", "yellow", "orange", "#ffe5b4"]
    
    apples = (fruit[0,0],fruit[1,0],fruit[2,0])
    bananas = (fruit[0,1],fruit[1,1],fruit[2,1])
    oranges = (fruit[0,2],fruit[1,2],fruit[2,2])
    peaches = (fruit[0,3],fruit[1,3],fruit[2,3])
    
    persons=("Farrah","Fred","Felicia" )
    weight_counts = {
    "apples": np.array([fruit[0,0],fruit[1,0],fruit[2,0]]),
    "bananas": np.array([fruit[0,1],fruit[1,1],fruit[2,1]]), 
    "oranges": np.array([fruit[0,2],fruit[1,2],fruit[2,2]]),
    "peaches": np.array([fruit[0,3],fruit[1,3],fruit[2,3]]),
    }
    bottom = np.zeros(3)
    i=0
    for boolean, weight_count in weight_counts.items():
        col=colors[i]
        i= i+1
        p = plt.bar(persons, weight_count, width= 0.5, color =col, label=boolean, bottom=bottom)
        bottom += weight_count
    plt.title("Number of Fruit per Person")
    plt.legend(loc="upper right")
    plt.ylabel("Quantity of Fruit")
    plt.show()
