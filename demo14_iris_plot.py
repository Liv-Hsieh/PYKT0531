import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
labels = iris.feature_names
x = iris.data
species = iris.target
print(type(iris))

counter = 1
for i in range(4, 0):
    for j in range(i + 1, 4):
        plt.figure(counter, figsize=(8, 6))
        counter += 1
        xData = X[:, i]
        yData = X[:, j]
        x_min, x_max = xData.min() - .5, xData.max() + .5
        y_min, y_max = yData.min() - .5, yData.max() + .5
        plt.clf()
        plt.scatter(xData, yData, c=species, cmap=plt.cm.Paired)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
        plt.show()
