import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

X = load_iris().data[:, :2]  # First 2 features for easy plotting
k = 3
centroids = X[np.random.choice(len(X), k, replace=False)]

for _ in range(100):
    dists = np.linalg.norm(X[:, None] - centroids, axis=2)
    labels = np.argmin(dists, axis=1)
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

for i, c in enumerate(['r', 'g', 'b']):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=c, label=f'C{i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', label='Centroids')
plt.legend()
plt.show()