#!/usr/bin/env python3
"""
Prject Hyperparameters tunings
By Ced+

"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    initilise the new class
    """
    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.Y_init = Y_init
        self.X_init = X_init

    def acquisition(self):
        """
        Donne le prochain test a effectuer de maniere a trouver
        le min ou le max d'une fonction
        """
        mu, sigma = self.gp.predict(self.X_s)

        # Calculer la meilleure valeur observée jusqu'à présent
        if self.minimize:
            mu_sample_opt = np.min(self.Y_init)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.Y_init)
            imp = mu - mu_sample_opt - self.xsi

        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        X_next= np.zeros((1,))
        Y_next= np.zeros((1,))
        X_opt = np.zeros((1,))
        Y_opt = np.zeros((1,))
        for i in range(iterations):
            
            try:
                old_X_next = X_next
                X_next, _ = self.acquisition()
                Y_next = self.f(X_next)
                if X_next == old_X_next:
                    break
                self.gp.update(X_next, Y_next)
            except:
                break
        X_opt = X_next
        Y_opt = Y_next
        return X_opt, Y_opt