from numpy import array
from sklearn.decomposition import PCA

#a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a = array([[10, 2, 6], [7, 54, 3], [0, 2, 1000]])
print(a)

pca = PCA(n_components=2)
pca.fit(a)
print('components=', pca.components_)
print("variance", pca.explained_variance_)
b = pca.transform(a)
print(b)