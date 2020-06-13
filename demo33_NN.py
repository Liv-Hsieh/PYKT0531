import numpy as np
from sklearn.neighbors import NearestNeighbors

x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

neighbors = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(x)

distances, indices = neighbors.kneighbors(x, return_distance=True)

print(f"distance={distances}")
print(f"indices={indices}")
print("nearest graph=")
print(neighbors.kneighbors_graph(x).toarray())