# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

dataset = load_iris()
X = dataset.data
target = dataset.target
# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create meanshift object
clt = DBSCAN(n_jobs=-1, min_samples=5, eps=0.3)

# Train model
model = clt.fit(X_std)

y_means = model.fit_predict(X_std)
labels = clt.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Visualizing the clusters
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'Traget')
# plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'cyan', label = 'Careless')
# plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[: ,0], kmeans.cluster_centers_[: ,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()