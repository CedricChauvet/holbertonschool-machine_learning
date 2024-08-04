#!/usr/bin/env python3
"""
Project Clusters
By Ced+
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    return dedrogramm with colors
    and the clss
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    # Utiliser dist comme seuil de couleur
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(Z, dist, criterion='distance')
    return clss
