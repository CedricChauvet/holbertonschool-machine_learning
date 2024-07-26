#!/usr/bin/env python3
"""
Project Dimensionality reduction
By Ced+
"""
import numpy as np


def pca(X, var=0.95):
    n = X.shape[0]  # number of data points
    d = X.shape[1]  # number of dimensions

    eigv_norm = list()
    W = np.ones((d,d), dtype= float) 

    # covariance matrix
    cov = X.T @ X
    # finding eigenvalues and eigenvectors
    eigv, W = np.linalg.eig(cov)
    
    # it appear eignenvalues a imaginary
    for i in range(d):
        eigv_norm.append(round(np.linalg.norm(eigv[i]),5))    
    
    # sorting elements
    zipW= list(zip(eigv_norm,W.T))
    sorted_zip =  sorted(zipW, key=lambda x:x[0], reverse = True)
    
    for j in range(d):
        print("sorted_zip", sorted_zip[j][0])
    
    
    # get nd wich is the domension reduction
    print("somme", sum(eigv_norm) )
    summation = 0
    i = 0
    threshold = sum(eigv_norm) * var
    while summation < threshold:
        summation += sorted_zip[i][0]
        i += 1
    
    
    nd = i + 1
    
    W_r = np.zeros((d,nd), dtype=float)
    
    # sorted zip a 2 dimension les deuxieme donne eigen
    for i in range(nd):
        W_r[:,i] = sorted_zip[i][1]

    # return -W_r