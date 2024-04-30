#!/usr/bin/env python3
"""
new class: NeuralNetwork, task  14: Evaluate Train NeuralNetwork
Write a class NeuralNetwork that defines a neural network with one
hidden layer performing binary classification

"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    class Neural network
    """
    def __init__(self, nx, nodes):

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nodes = nodes

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__A1 = np.array(np.dot(X.T, self.__W1.T) + self.__b1.T)
        self.__A1 = np.array(1 / (1 + np.exp(-self.__A1)))
        self.__A2 = np.array(np.dot(self.__A1, self.__W2.T) + self.__b2)
        self.__A2 = np.array(1 / (1 + np.exp(-self.__A2))).T
        self.__A1 = self.__A1.T
        # Carreful on shapes self.__A1 and A2 are transposed
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calcul of the cost function"""

        ni = Y.shape[1]
        """ Need an intermediary part,
          calcul of the loss function"""
        # first calcul loss function
        loss = -(Y * np.log(A) + (1-Y) * np.log(1.0000001 - A))
        # the cost
        cost = (1 / ni) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron's network predictions"""
        # forward propagation on data X
        A1, A2 = self.forward_prop(X)

        # my cost calcul
        cost = self.cost(Y, A2)

        # nympy where, put 0 if forward(X) < 0.5, 1 else
        Btest = np.where(A2 >= 0.5, 1, 0)
        return Btest, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        # taille des exemples
        m = X.shape[1]

        dZee2 = A2 - Y
        # calcul des nouvelles valeurs de W2 et b2 par gradient descent
        dW2 = (1 / m) * np.dot(dZee2, A1.T)
        db2 = (1 / m) * np.sum(dZee2, axis=1, keepdims=True)

        dZee1 = np.dot(self.__W2.T, dZee2) * A1 * (1. - A1)
        dW1 = 1 / m * np.dot(dZee1, X.T)
        dB1 = 1 / m * np.sum(dZee1, axis=1, keepdims=True)

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * dB1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha / m * np.sum(dZee2, axis=1,
                                                   keepdims=True)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ fontion d'entrainement ameliorée
        Trains the neural network
        """
        # using variable for plotting
        plot_cost = np.array([])
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        elif iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        elif alpha < 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A, cost = self.evaluate(X, Y)

            plot_cost = np.append(plot_cost, cost)
            # verbose mode
            if verbose:
                print(f"Cost after {i} iterations: {cost}")

            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha=alpha)

        A, cost = self.evaluate(X, Y)

        # Visual Mode
        if graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            elif step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

            x = np.arange(0, iterations, step)
            plt.plot(x, plot_cost[x])
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return A, cost
