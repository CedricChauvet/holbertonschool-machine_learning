#!/usr/bin/env python3
"""
documentation création d'une classe neuro task 4: Evaluate Neuron

goal:  Write a class Neuron that defines a single neuron
performing binary classification (
"""
import numpy as np


class Neuron:
    """ adding a cost function, and a logarithmic loss
      calcul,evaluation, gradient descent """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.nx = nx
        # private attributsS
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def set_A(self, p):
        """Creation of a setter for A,  ***no setter decorator*** """
        self.__A = p

    def set_W(self, p):
        """Creation of a setter for W,  ***no setter decorator*** """
        self.__W = p

    def set_b(self, p):
        """Creation of a setter for b,  ***no setter decorator*** """
        self.__b = p

    def forward_prop(self, X):
        """ Take the Xnx imnut of a neuron
        the dot X with the weight and biases for
         forward propagation """

        # aprroximation with transpose but works
        forward_var = np.array(np.dot(X.T, self.__W.T) + self.__b)
        activation = np.array(1 / (1 + np.exp(-forward_var))).T
        self.set_A(activation)

        return self.__A

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
        """Evaluates the neuron's predictions"""
        # forward propagation on data X
        Atest = self.forward_prop(X)

        # my cost calcul
        cost = self.cost(Y, Atest)

        # nympy where, put 0 if forward(X) < 0.5, 1 else
        Btest = np.where(Atest >= 0.5, 1, 0)

        return Btest, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
       """ on a Zee= w.T *x +b
            on a besoin de caluler le gradient de Zee
            on sait que le gradient (grad = a -Y)
            la formule de a est la sigmoide de z
       """ 
       #nombre de parametres d'entrée
       ni = X.shape[0]
       # taille des exemples
       m = X.shape[1]
       
             
       # method gradient descent
       #Zee = np.dot(self.__W, X) + self.__b 
       
       # a =  np.array(1 / (1 + np.exp(-Zee)))
       dZee = A - Y
       #print("dZee" , dZee.shape) 
       #print("X", X.shape)
       #print("a", a.shape) 
       #print("W", self.__W.shape)
       #print("Zee", Zee.shape)
       #print("shapedot", (self.__W.T - alpha * np.dot(X, dZee.T)).shape)
       
       # calcul des nouvelles valeurs de Wi et w par gradient descent
       dw = 1/ m * np.dot(X, dZee.T)
       w = self.__W.T - alpha * dw
       self.set_W(w.T)

       #calcul de b par gradient descent
       self.set_b(self.__b - np.sum(alpha*dZee) / m)