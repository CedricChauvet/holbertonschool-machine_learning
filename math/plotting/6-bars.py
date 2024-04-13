#!/usr/bin/env python3
"""
Task 6, project plotting: Stacking bars
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """" This is a documentation """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
    persons = ("Farrah", "Fred", "Felicia")

    weight_counts = {
        "apples": np.array([fruit[0, 0] fruit[0, 1], fruit[0, 2]]),
        "bananas": np.array([fruit[1, 0], fruit[1, 1], fruit[1, 2]]),
        "oranges": np.array([fruit[2, 0], fruit[2, 1], fruit[2, 2]]),
        "peaches": np.array([fruit[3, 0], fruit[3, 1], fruit[3, 2]])
    }

    bottom = np.zeros(3)
    i = 0
    for boolean, weight_count in weight_counts.items():
        col = colors[i]
        i = i+1
        p = plt.bar(persons, weight_count, width=0.5,
                    color=col, label=boolean, bottom=bottom)
        bottom += weight_count
    plt.title("Number of Fruit per Person")
    plt.legend(loc="upper right")
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.show()
