import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

x = np.r_[np.random.randn(5000, 2) + [2, 2],
          np.random.randn(5000, 2) + [-2, -2],
          np.random.randn(5000, 2) + [2, -4],
          np.random.randn(5000, 2) + [0, 0]]

inertias = []
for k in range(1,21):
    kmeans = KMeans(n_clusters=k).fit(x)
    inertias.append(kmeans.inertia_)

print(inertias)
plt.plot(range(1, 21), inertias)
plt.show()