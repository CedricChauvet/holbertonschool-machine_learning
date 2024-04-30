#!/usr/bin/env python3
"""Write a function def one_hot_encode(Y, classes): that converts
 a numeric label vector into a one-hot matrix and reverse
 Task 26: . Persistence is Key
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """ define a new class with private attributes"""
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.layers = layers
        self.__weights = dict()
        self.__L = len(layers)
        self.__cache = dict()

        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights['W1'] = np.random.randn(
                        layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W{}'.format(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights['b{}'.format(i + 1)] = np.zeros((layers[i], 1))

        # He et al. method:
        # w=np.random.randn(layer_size[l],layer_size[l-1])*np.sqrt(2/layer_size[l-1])

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """ forward propagation in a DNN, be careful of Matrix dimensions"""
        for i in range(self.__L + 1):
            if i == 0:
                self.__cache['A0'] = X
            else:
                z = np.dot(self.__weights['W{}'.format(
                    i)], self.__cache['A{}'.format(i - 1)]
                    ) + self.__weights['b{}'.format(i)]
                sigmoid_z = 1 / (1 + np.exp(-z))
                self.__cache['A{}'.format(i)] = sigmoid_z
        # retourne le dernier Layer, puis l'ensemble des
        # données dans cache, toutes les activations
        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calcul of the cost function in a DNN"""

        ni = Y.shape[1]
        """ Need an intermediary part,
          calcul of the loss function"""

        # first calcul loss function
        loss = -(Y * np.log(A) + (1-Y) * np.log(1.0000001 - A))
        # the cost
        cost = (1 / ni) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the deep neuron's network predictions"""
        # forward propagation on data X
        A, self.__cache = self.forward_prop(X)

        # my cost calcul
        cost = self.cost(Y, A)

        # nympy where, put 0 if forward(X) < 0.5, 1 else
        Btest = np.where(A >= 0.5, 1, 0)
        return Btest, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on a DNN"""
        # taille des exemples
        m = Y.shape[1]
        dZee = dict()
        dB = dict()
        dW = dict()

        dZee['Z{}'.format(self.__L)] = self.cache['A{}'.format(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            dSigmoid = cache['A{}'.format(i - 1)
                             ] * (1 - cache['A{}'.format(i - 1)])

            # calcul differentiel
            dZee[f"Z{i-1}"] = np.dot(
                self.__weights[f"W{i}"].T, dZee[f"Z{i}"]) * dSigmoid
            dB[f"b{i}"] = 1 / m * np.sum(dZee[f"Z{i}"], axis=1, keepdims=True)
            dW[f"W{i}"] = 1 / m * np.dot(dZee[f"Z{i}"], cache[f"A{i-1}"].T)

            # MAJ des parametres W et b
            self.__weights[f"b{i}"] = self.__weights[
                f"b{i}"] - alpha * dB[f"b{i}"]
            self.__weights[f"W{i}"] = self.__weights[
                f"W{i}"] - alpha * dW[f"W{i}"]

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ fontion d'entrainement
        Trains the DNN
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
            Aact, cost = self.evaluate(X, Y)

            plot_cost = np.append(plot_cost, cost)
            # verbose mode
            if verbose:
                print(f"Cost after {i} iterations: {cost}")

            self.gradient_descent(Y, self.__cache, alpha)
        # last evaluation to MAJ weights and biaises
        Aact, cost = self.evaluate(X, Y)

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
        return Aact, cost

    def save(self, filename = None):
        """ save the instance"""
        if filename == None:
            return None
        if not filename.lower().endswith(".pkl"):
            filename += ".pkl"
        file = open(filename, 'wb')
        pickle.dump(self, file)
        file.close()

    @staticmethod
    def load(filename):
        """load the instance"""
        file = open(filename, 'rb')
        return pickle.load(file)
