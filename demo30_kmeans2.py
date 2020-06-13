import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

x = np.r_[np.random.randn(100, 2) + [2, 2],
          np.random.randn(100, 2) + [-2, -2],
          np.random.randn(100, 2) + [2, -4]]

[plt.scatter(e[0], e[1], c='black', s=7) for e in x]

k = 3
C_x = np.random.uniform(np.min(x[:, 0]), np.max(x[:, 0]), size=k)
C_y = np.random.uniform(np.min(x[:, 1]), np.max(x[:, 1]), size=k)
print(type(C_x), C_x)
print(type(C_y), C_y)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(type(C), f'print C :{C}')
plt.scatter(C_x, C_y, marker="*", s=200, c='#005599')
plt.show()


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# print(dist(np.array([[0, 0]]), np.array([[2, 2]])))

C_old = np.zeros(C.shape)
clusters = np.zeros(len(x))
delta = dist(C, C_old, None)
print(f"delta={delta}")


def plot_kmean(current_cluster, delta):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots()
    for i in range(k):
        points = np.array([x[j] for j in range(len(x)) if current_cluster[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker="*", s=200, c="#005599")
    plt.title(f"delta for C will be:{delta:.4f} ")
    plt.plot()
    plt.show()


while delta != 0:
    print("start a new iteration")
    for i in range(len(x)):
        distances = dist(x[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    for i in range(k):
        points = [x[j] for j in range(len(x)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    delta = dist(C, C_old, None)
    print(f"delta={delta}")
    plot_kmean(clusters, delta)
