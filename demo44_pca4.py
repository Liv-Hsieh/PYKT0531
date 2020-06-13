import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,svm
from sklearn.decomposition import PCA

iris = datasets.load_iris()

pca = PCA(n_components=2)
data = pca.fit(iris.data).transform(iris.data)
print(data.shape)
print(data[:5, ])
datamax = data.max(axis=0) + 1
datamin = data.min(axis=0) - 1
n = 200
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   (np.linspace(datamin[1], datamax[1], n)))
svc = svm.SVC()
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
plt.contour(X, Y, Z.reshape(X.shape), levels=[-0.5, 0.5, 1.5, 2.5], colors=['r', 'g', 'b'])

for i, c in zip([0, 1, 2], ['r', 'g', 'b']):
    d = data[iris.target == i]
    plt.scatter(d[:, 0], d[:, 1], c=c)
plt.show()