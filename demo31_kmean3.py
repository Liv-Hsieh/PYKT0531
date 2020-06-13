import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

x = np.r_[np.random.randn(100, 2) + [2, 2],
          np.random.randn(100, 2) + [-2, -2],
          np.random.randn(100, 2) + [2, -4]]
k = 3
kmeans = KMeans(n_clusters=k).fit(x)
print(kmeans.cluster_centers_)
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
markers = ['s', '^', 'o', '.', '*', 'v', 'x']

for i in range(k):
    currentX = x[kmeans.labels_ == i]
    print(f'print currentX KMeans Label = {x[kmeans.labels_]}')
    plt.scatter(currentX[:, 0], currentX[:, 1], c=colors[i], marker=markers[i])
    print(f"#{i} clsuter has element:{currentX.size}")

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='*', s=200, c='#C0FFEE')
print(f"current cost:{kmeans.inertia_}")
plt.show()