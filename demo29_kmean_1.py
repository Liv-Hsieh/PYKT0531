from sklearn.cluster import KMeans
import numpy as np

x = np.array([[1, 0], [0, 1], [1, 2], [1, 4], [1, 6], [4, 2], [4, 4], [4, 0], [4, 6], [4, 7], [1000, 1000],
[-2, -2], [-4, -2], [-1, -3]])
for i in range(0, 10):
    kmeans = KMeans(n_clusters=2, n_init=20).fit(x)
    for i in range(0, len(x)):
        print(f"{x[i]} is belonging to cluster#{kmeans.labels_[i]}")
    print(f"center for kmean = {kmeans.cluster_centers_}")
    print(f"kmeans inertia={kmeans.inertia_}")
