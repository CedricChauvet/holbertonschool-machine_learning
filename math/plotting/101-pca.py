#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

data =  np.load("data.npy")
labels = np.load("labels.npy")

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)


fig = plt.figure()
Axes3D = fig.add_subplot(111, projection='3d')
Axes3D.scatter(xs=pca_data[:, 0], ys=pca_data[:, 1], zs=pca_data[:, 2],zdir='z', c=labels)
Axes3D.set_xlabel("U1")
Axes3D.set_ylabel("U2")
Axes3D.set_zlabel("U3")
Axes3D.set_title("PCA of Iris Dataset")
plt.show()